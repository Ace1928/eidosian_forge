from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export
Layer that applies an update to the cost function based input activity.

    Args:
      l1: L1 regularization factor (positive float).
      l2: L2 regularization factor (positive float).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    