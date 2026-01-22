import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import utils
from keras.src.applications import imagenet_utils
from keras.src.engine import sequential
from keras.src.engine import training as training_lib
from tensorflow.python.util.tf_export import keras_export
def PreStem(name=None):
    """Normalizes inputs with ImageNet-1k mean and std.

    Args:
      name (str): Name prefix.

    Returns:
      A presemt function.
    """
    if name is None:
        name = 'prestem' + str(backend.get_uid('prestem'))

    def apply(x):
        x = layers.Normalization(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2], name=name + '_prestem_normalization')(x)
        return x
    return apply