import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).

    Args:
      inputs: Input tensor of shape `(rows, cols, 3)` (with `channels_last`
        data format) or (3, rows, cols) (with `channels_first` data format).
        It should have exactly 3 inputs channels, and width and height should
        be no smaller than 32. E.g. `(224, 224, 3)` would be one valid value.
      filters: Integer, the dimensionality of the output space (i.e. the
        number of output filters in the convolution).
      alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
      kernel: An integer or tuple/list of 2 integers, specifying the width and
        height of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any `dilation_rate`
        value != 1. # Input shape
      4D tensor with shape: `(samples, channels, rows, cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, rows, cols, channels)` if
        data_format='channels_last'. # Output shape
      4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have
        changed due to stride.

    Returns:
      Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1')(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6.0, name='conv1_relu')(x)