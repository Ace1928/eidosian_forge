from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ray import train, tune
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining
Train keras CNN on the CIFAR10 small images dataset.

The model comes from: https://zhuanlan.zhihu.com/p/29214791,
and it gets to about 87% validation accuracy in 100 epochs.

Note that the script requires a machine with 4 GPUs. You
can set {"gpu": 0} to use CPUs for training, although
it is less efficient.
