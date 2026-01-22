import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=(1, 1), block_id=None):
    """Adds 2 blocks of [relu-separable conv-batchnorm].

    Args:
        ip: Input tensor
        filters: Number of output filters per layer
        kernel_size: Kernel size of separable convolutions
        strides: Strided convolution for downsampling
        block_id: String block_id

    Returns:
        A Keras tensor
    """
    channel_dim = 1 if backend.image_data_format() == 'channels_first' else -1
    with backend.name_scope(f'separable_conv_block_{block_id}'):
        x = layers.Activation('relu')(ip)
        if strides == (2, 2):
            x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, kernel_size), name=f'separable_conv_1_pad_{block_id}')(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = layers.SeparableConv2D(filters, kernel_size, strides=strides, name=f'separable_conv_1_{block_id}', padding=conv_pad, use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=0.001, name=f'separable_conv_1_bn_{block_id}')(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, kernel_size, name=f'separable_conv_2_{block_id}', padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=0.001, name=f'separable_conv_2_bn_{block_id}')(x)
    return x