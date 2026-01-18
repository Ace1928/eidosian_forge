import json
import warnings
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src.utils import data_utils
from tensorflow.python.util.tf_export import keras_export
validates that the classifer_activation is compatible with the weights.

    Args:
      classifier_activation: str or callable activation function
      weights: The pretrained weights to load.

    Raises:
      ValueError: if an activation other than `None` or `softmax` are used with
        pretrained weights.
    