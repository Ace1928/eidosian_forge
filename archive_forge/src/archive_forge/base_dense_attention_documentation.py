import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.utils import control_flow_util
from tensorflow.python.util.tf_export import keras_export
Validates arguments of the call method.