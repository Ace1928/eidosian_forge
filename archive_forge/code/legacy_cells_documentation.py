from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import warnings
import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import backend
from keras.src import initializers
from keras.src.engine import base_layer_utils
from keras.src.engine import input_spec
from keras.src.legacy_tf_layers import base as base_layer
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util.tf_export import tf_export
Run this multi-layer cell on inputs, starting from state.