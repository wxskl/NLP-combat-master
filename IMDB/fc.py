import torch
import random
import time
import spacy
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 利用随机数种子来使结果可以复现
SEED = 1234
# 为CPU/GPU设置种子用于生成随机数，以使得结果是确定的
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# 采用确定性卷积,以便重现结果
torch.backends.cudnn.deterministic = True

# 1）下载 imbd 数据，train & test
# 也可以使用 nltk 的 punkt
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

print("\ndowning IMDB data...")
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print('finished...')

print('***拿一条数据看看***')
print(vars(train_data.examples[0]))


# 2) 数据切分 train valid
print("\n切分数据 train valid data...")
train_data, valid_data = train_data.split(random_state=random.seed(SEED), split_ratio=0.8)
print("Number of training examples: ", len(train_data))
print("Number of validation examples: ", len(valid_data))
print("Number of testing examples: ", len(test_data))
print('finished...')

# 3) build vocab， 并定义max_size  & glove
print("\nbuild vocab， 并定义max_size  & glove...")
MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.300d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(test_data)
print(f'Vocabulary size: {len(TEXT.vocab)}')
print(f'Number of classes: {len(LABEL.vocab)}')

print('***************')
print("TEXT.vocab.freqs.most_common(20)", TEXT.vocab.freqs.most_common(20))
print('---------------')
print("TEXT.vocab.itos[:10]", TEXT.vocab.itos[:10])
print('***************')
print('finished')

# 4) 创建 iterator batch-examples
print("\n创建 iterator batch-examples...")
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)
print('finished')


## 5) Word Averaging 模型
class WordAvg(nn.Module):
    def __init__(self, vocab_size, embeding_size, output_size, pad_idx):
        super(WordAvg, self).__init__()
        self.embed = nn.Embedding(vocab_size, embeding_size, padding_idx=pad_idx)
        self.fc = nn.Linear(embeding_size, output_size)

    def forward(self, text):
        embed = self.embed(text)  # sen_length, batch_size, embed_size
        embed = embed.permute(1, 0, 2)  # batch_size, sen_length, embed_size
        pooled = F.avg_pool2d(embed, (embed.shape[1], 1)).squeeze(1)  # batch_size, embed_size
        return self.fc(pooled)


# 6) 定义模型参数
print("\n定义模型参数 & 创建模型...")
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
OUTPUT_DIM = 1  # (积极，消极)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # string to index, pad的序号是啥
model = WordAvg(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
print('finished')

# 7) 载入预训练的词向量 初始化 UNK, PAD 矩阵中的值为0
pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# 8）训练模型
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()  # 多分类的cross entropy
model = model.to(device)
# model = nn.DataParallel(model, device_ids=[1, 2, 3])  # multi-GPU
criterion = criterion.to(device)


# 计算准确率
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))  # 把概率的结果 四舍五入
    correct = (rounded_preds == y).float()  # True False -> 转为 1， 0
    acc = correct.sum() / len(correct)
    return acc


# train
def train(model, iterator, optimizer, criterion):
    """
    :param model:  传入模型
    :param iterator: 传入多个 batch 组成的输入
    :param optimizer: 传入优化算法 optimizer
    :param criterion: 传入计算loss的方法
    :return:
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()  # 注意切换模式

    for batch in iterator:  # 有多少个batch
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)  # 在第1个维度上 去除维度=1
        loss = criterion(predictions, batch.label)  # 计算loss，用于backward()
        acc = binary_accuracy(predictions, batch.label)  # 计算acc，看下这次batch的准确度
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 取出loss中的值
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            prediction = model(batch.text).squeeze(1)
            loss = criterion(prediction, batch.label)
            acc = binary_accuracy(prediction, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(end_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10

best_valid_loss = float('inf')

print("\n训练开始...")
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')
        print('save ok, valid_loss {}'.format(valid_loss))

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


print('ok')
print("\n预测开始...")

nlp = spacy.load('en')  # 这就是个tokenizer


def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    print('tokenized :', tokenized)
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    print('indexed :', indexed)
    tensor = torch.LongTensor(indexed).to(device)
    print('tensor :', tensor)
    tensor = tensor.unsqueeze(1)
    print('tensor after unsqueeze(1) :', tensor)
    prediction = torch.sigmoid(model(tensor))
    print("prediction.item() :", prediction)
    return prediction.item()


sen = "Rick and Morty is awesome"
print('\n预测 sen = ', sen)
print('预测 结果:', predict_sentiment(sen))


