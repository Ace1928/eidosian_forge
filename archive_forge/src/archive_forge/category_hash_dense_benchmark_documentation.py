import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import hashing
from keras.src.layers.preprocessing.benchmarks import (
from tensorflow.python.eager.def_function import (
Benchmark the layer forward pass.