import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def transition_block(x, reduction, name):
    """A transition block.

    Args:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.

    Returns:
      output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x