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
def ConvNeXtBlock(projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-06, name=None):
    """ConvNeXt block.

    References:
    - https://arxiv.org/abs/2201.03545
    - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Notes:
      In the original ConvNeXt implementation (linked above), the authors use
      `Dense` layers for pointwise convolutions for increased efficiency.
      Following that, this implementation also uses the same.

    Args:
      projection_dim (int): Number of filters for convolution layers. In the
        ConvNeXt paper, this is referred to as projection dimension.
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
      layer_scale_init_value (float): Layer scale value. Should be a small float
        number.
      name: name to path to the keras layer.

    Returns:
      A function representing a ConvNeXtBlock block.
    """
    if name is None:
        name = 'prestem' + str(backend.get_uid('prestem'))

    def apply(inputs):
        x = inputs
        x = layers.Conv2D(filters=projection_dim, kernel_size=7, padding='same', groups=projection_dim, name=name + '_depthwise_conv')(x)
        x = layers.LayerNormalization(epsilon=1e-06, name=name + '_layernorm')(x)
        x = layers.Dense(4 * projection_dim, name=name + '_pointwise_conv_1')(x)
        x = layers.Activation('gelu', name=name + '_gelu')(x)
        x = layers.Dense(projection_dim, name=name + '_pointwise_conv_2')(x)
        if layer_scale_init_value is not None:
            x = LayerScale(layer_scale_init_value, projection_dim, name=name + '_layer_scale')(x)
        if drop_path_rate:
            layer = StochasticDepth(drop_path_rate, name=name + '_stochastic_depth')
        else:
            layer = layers.Activation('linear', name=name + '_identity')
        return inputs + layer(x)
    return apply