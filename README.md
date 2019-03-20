<div align='center'>
    <img src= 'https://github.com/Mingqi-Yuan/ADMP/blob/master/example/pulseai_logo.png'>
</div>
<h1 align="center">
    藏文手写体数字数据集
    <br>
    Tibetan Handwritten Digital Dataset
</h1>

# **背景描述**
MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST). 训练集由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员。自MNIST数据集建立以来，被广泛地应用于检验各种机器学习算法，测试各种模型，为机器学习的发展做出了不可磨灭的贡献，其当之无愧为历史上最伟大的数据集之一。在一次会议上，我无意间看到了一位藏族伙伴的笔记本上写着一些奇特的符号，好奇心驱使我去了解这些符号的意义，我的伙伴告诉我，这些是藏文当中的数字，这对于从小使用阿拉伯数字的我十分惊讶，这些奇特的符号竟有如此特殊的含义！我当即产生了一个想法，能不能让计算机也能识别这些数字呢？这个想法得到了大家的一致认可，于是我们开始模仿MNIST来制作这些数据，由于对藏文的不熟悉，一开始的工作十分艰难，直到取得了藏学研究院同学的帮助，才使得制作工作顺利完成。历时1个月，超过300次反复筛选，最终得到17768张高清藏文手写体数字图像，形成了**TibetanMNIST**数据集。我和我的团队为其而骄傲，因为它不仅仅是我们自行制作的第一个数据集，更是第一个藏文手写数字的图像数据集！藏文手写数字和阿拉伯数字一样，在藏文中是一个独立的个体，具有笔画简单，便于识别等优良特性。经过反复地商议，我们决定将其完全开源，供所有的开发者自由使用，使其能发挥最大的价值！为了方便大家使用，我们将数据制作成了TFRecords以及npz文件格式【文件顺序未打乱】，使其便于读取，能很好地配合现有机器学习框架使用，当然，如果你觉得它还可以做的更好，你也可以自行DIY，我们将分割后的原始图像也上传到了科赛平台上，你可以将其做成你喜欢的任何数据格式，并创建各种有趣的项目。我和我的团队衷心地希望你能在使用它的过程获得乐趣！
最后，十分感谢科赛网提供的平台，为数据的维护和推广提供了极大的便利！能让更多人看到藏文数字和原创数据的美，就是我们最大的收获！
   
**——袁明奇、才让先木等
PulseAI**

2018年11月27日

# **数据下载**
[点此下载数据集](https://www.kesci.com/urls/74bacce8)

# **数据说明**
*  **数据文化**
藏区按方言划分为卫藏、康巴、安多三大藏区，东接汉地九州。藏区有典：“法域卫藏、人域康巴、马域安多”即“卫藏的宗教、康巴的人、安多的马”。
而藏文主要有楷体和形体两种文字。我们本次的TibetanMNIST正是形体藏文中的数字，也就是图片中连笔书写更加简洁的那种~



![Image Name](https://cdn.kesci.com/upload/image/pixx2ees7d.png?imageView2/0/w/320/h/320)





 * **文件列表**

- TibetanMNIST.tfrecords **（每张图像存储为28x28x3的三通道图像矩阵）**
- TibetanMNIST.npz **（每张图像存储为28x28的单通道图像矩阵）**
- TibetanMNIST.zip
 
**TibetanMnist.zip文件为原始图像文件，图像文件名的第一个数字为数字类别标签，第二个数字为数字所在纸张标签，第三个数字为纸张标签内的数字序列。**

 
* **数据特征及属性**


| 数据集名称 | 数据类型 |原始图像宽|原始图像高|TFRecords图像宽|TFRecords图像高|通道数|位深度|值缺失| 实例数 |相关任务|
|:--------:|:--------:|:--------:|:-:|:--------:|:--------:|:--:|:-:|:-:|:-:|:-:|
| TibetanMnist  | 图像数据|350|350|28|28|单通道|8| N/A    |17768  |分类任务|
* **数据分布**

|0|1|2|3|4|5|6|7|8|9|总计|
|:------:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:--:|:---:|
|1996|2134|1872|1816|1967|1350|1618|1302|1642|2071|17768|

* **藏文数字与阿拉伯数字对照表**

|0|1|2|3|4|5|6|7|8|9|
|------|----|--|--|--|--|--|--|--|--|
|**༠**|**༡**|**༢**|**༣**|**༤**|**༥**|**༦**|**༧**|**༨**|**༩**|

* **数据示例**


![Image Name](https://cdn.kesci.com/upload/image/pixx4l3j4p.jpg?imageView2/0/w/320/h/320)



# **数据处理** 
我们使用Keras建立了一个BP网络来训练这批数据，以下为网络代码：
```python
from tensorflow import keras
from keras import utils
import numpy as np

data = np.load('E:/TibetanMNIST.npz')
x_train = data['image'].reshape(17768, 784)
y_train = utils.to_categorical(data['label'], 10)

# create model
model = keras.Sequential()
model.add(keras.layers.Dense(784, input_dim=784, kernel_initializer='normal', activation= 'tanh'))
model.add(keras.layers.Dense(512, kernel_initializer='normal', activation= 'tanh'))
model.add(keras.layers.Dense(512, kernel_initializer='normal', activation= 'tanh'))
model.add(keras.layers.Dense(10, kernel_initializer='normal', activation= 'softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, epochs=10, batch_size=200, verbose=1)
```

对网络结构进行可视化：

![Image Name](https://cdn.kesci.com/upload/image/piw2ylt96x.png?imageView2/0/w/640/h/640)

在训练10个世代后，精度稳定在94%左右：
```
  200/17768 [..............................] - ETA: 0s - loss: 0.1813 - acc: 0.9400
 2400/17768 [===>..........................] - ETA: 0s - loss: 0.1629 - acc: 0.9421
 4200/17768 [======>.......................] - ETA: 0s - loss: 0.1541 - acc: 0.9452
 6600/17768 [==========>...................] - ETA: 0s - loss: 0.1603 - acc: 0.9432
 9000/17768 [==============>...............] - ETA: 0s - loss: 0.1614 - acc: 0.9427
11000/17768 [=================>............] - ETA: 0s - loss: 0.1640 - acc: 0.9408
13200/17768 [=====================>........] - ETA: 0s - loss: 0.1632 - acc: 0.9402
15600/17768 [=========================>....] - ETA: 0s - loss: 0.1662 - acc: 0.9392
17600/17768 [============================>.] - ETA: 0s - loss: 0.1698 - acc: 0.9387
17768/17768 [==============================] - 0s 27us/step - loss: 0.1698 - acc: 0.9387
```


# **数据使用**
* **TFReords**文件使用


```python
import tensorflow as tf

def _parse_function(example_proto):
  features = {'label':tf.FixedLenFeature([], tf.int64),
              'img_raw':tf.FixedLenFeature([], tf.string)}
  
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
  img = tf.reshape(img, [28, 28, 3])
    # 在流中抛出img张量和label张量
  img = tf.cast(img, tf.float32) / 255
  label = tf.cast(parsed_features['label'], tf.int32)
  return img, label

filenames = ["要读取的文件序列"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
# 创建单次迭代器
iterator = dataset.make_one_shot_iterator()
# 读取图像数据、标签值
image, label = iterator.get_next()

```


* ** NPZ**文件读取


```python
import numpy as np

data = np.load('文件')
x_train = data['image'].reshape(17768, 784)
y_train = utils.to_categorical(data['label'], 10)
```






# **数据来源**
PulseAI

# **使用注意**
本数据集版权归PulseAI所有，使用该数据请务必注明数据出处，否则我们将追究相应的法律责任！
