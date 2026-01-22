import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import models
from keras.src.applications import imagenet_utils
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        prefix = f'expanded_conv_{block_id}/'
        x = layers.Conv2D(_depth(infilters * expansion), kernel_size=1, padding='same', use_bias=False, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis, epsilon=0.001, momentum=0.999, name=prefix + 'expand/BatchNorm')(x)
        x = activation(x)
    if stride == 2:
        x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, kernel_size), name=prefix + 'depthwise/pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same' if stride == 1 else 'valid', use_bias=False, name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=0.001, momentum=0.999, name=prefix + 'depthwise/BatchNorm')(x)
    x = activation(x)
    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=0.001, momentum=0.999, name=prefix + 'project/BatchNorm')(x)
    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x