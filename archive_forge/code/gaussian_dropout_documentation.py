import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    Args:
      rate: Float, drop probability (as with `Dropout`).
        The multiplicative noise will have
        standard deviation `sqrt(rate / (1 - rate))`.
      seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    