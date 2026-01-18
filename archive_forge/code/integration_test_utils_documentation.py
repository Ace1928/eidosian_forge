import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src.datasets import mnist
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.utils import np_utils
Builds a Sequential CNN model to recognize MNIST digits.