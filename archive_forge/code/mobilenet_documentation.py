import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    Args:
      inputs: Input tensor of shape `(rows, cols, channels)` (with
        `channels_last` data format) or (channels, rows, cols) (with
        `channels_first` data format).
      pointwise_conv_filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the pointwise convolution).
      alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
      depth_multiplier: The number of depthwise convolution output channels
        for each input channel. The total number of depthwise convolution
        output channels will be equal to `filters_in * depth_multiplier`.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any `dilation_rate`
        value != 1.
      block_id: Integer, a unique identification designating the block number.
        # Input shape
      4D tensor with shape: `(batch, channels, rows, cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(batch, rows, cols, channels)` if
        data_format='channels_last'. # Output shape
      4D tensor with shape: `(batch, filters, new_rows, new_cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(batch, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have
        changed due to stride.

    Returns:
      Output tensor of block.
    