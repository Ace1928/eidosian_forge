import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _reduction_a_cell(ip, p, filters, block_id=None):
    """Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).

    Args:
      ip: Input tensor `x`
      p: Input tensor `p`
      filters: Number of output filters
      block_id: String block_id

    Returns:
      A Keras tensor
    """
    channel_dim = 1 if backend.image_data_format() == 'channels_first' else -1
    with backend.name_scope(f'reduction_A_block_{block_id}'):
        p = _adjust_block(p, ip, filters, block_id)
        h = layers.Activation('relu')(ip)
        h = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=f'reduction_conv_1_{block_id}', use_bias=False, kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=0.001, name=f'reduction_bn_1_{block_id}')(h)
        h3 = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(h, 3), name=f'reduction_pad_1_{block_id}')(h)
        with backend.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2), block_id=f'reduction_left1_{block_id}')
            x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), block_id=f'reduction_right1_{block_id}')
            x1 = layers.add([x1_1, x1_2], name=f'reduction_add_1_{block_id}')
        with backend.name_scope('block_2'):
            x2_1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name=f'reduction_left2_{block_id}')(h3)
            x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), block_id=f'reduction_right2_{block_id}')
            x2 = layers.add([x2_1, x2_2], name=f'reduction_add_2_{block_id}')
        with backend.name_scope('block_3'):
            x3_1 = layers.AveragePooling2D((3, 3), strides=(2, 2), padding='valid', name=f'reduction_left3_{block_id}')(h3)
            x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2), block_id=f'reduction_right3_{block_id}')
            x3 = layers.add([x3_1, x3_2], name=f'reduction_add3_{block_id}')
        with backend.name_scope('block_4'):
            x4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=f'reduction_left4_{block_id}')(x1)
            x4 = layers.add([x2, x4])
        with backend.name_scope('block_5'):
            x5_1 = _separable_conv_block(x1, filters, (3, 3), block_id=f'reduction_left4_{block_id}')
            x5_2 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name=f'reduction_right5_{block_id}')(h3)
            x5 = layers.add([x5_1, x5_2], name=f'reduction_add4_{block_id}')
        x = layers.concatenate([x2, x3, x4, x5], axis=channel_dim, name=f'reduction_concat_{block_id}')
        return (x, ip)