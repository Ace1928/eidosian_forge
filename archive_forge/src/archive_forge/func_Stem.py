import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def Stem(name=None):
    """Implementation of RegNet stem.

    (Common to all model variants)
    Args:
      name: name prefix

    Returns:
      Output tensor of the Stem
    """
    if name is None:
        name = 'stem' + str(backend.get_uid('stem'))

    def apply(x):
        x = layers.Conv2D(32, (3, 3), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal', name=name + '_stem_conv')(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-05, name=name + '_stem_bn')(x)
        x = layers.ReLU(name=name + '_stem_relu')(x)
        return x
    return apply