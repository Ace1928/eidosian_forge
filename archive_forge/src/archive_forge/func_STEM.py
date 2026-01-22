import sys
from typing import Callable
from typing import Dict
from typing import List
from typing import Union
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def STEM(bn_momentum: float=0.0, bn_epsilon: float=1e-05, activation: str='relu', name=None):
    """ResNet-D type STEM block."""
    if name is None:
        counter = backend.get_uid('stem_')
        name = f'stem_{counter}'

    def apply(inputs):
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = Conv2DFixedPadding(filters=32, kernel_size=3, strides=2, name=name + '_stem_conv_1')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, name=name + '_stem_batch_norm_1')(x)
        x = layers.Activation(activation, name=name + '_stem_act_1')(x)
        x = Conv2DFixedPadding(filters=32, kernel_size=3, strides=1, name=name + '_stem_conv_2')(x)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, name=name + '_stem_batch_norm_2')(x)
        x = layers.Activation(activation, name=name + '_stem_act_2')(x)
        x = Conv2DFixedPadding(filters=64, kernel_size=3, strides=1, name=name + '_stem_conv_3')(x)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, name=name + '_stem_batch_norm_3')(x)
        x = layers.Activation(activation, name=name + '_stem_act_3')(x)
        x = Conv2DFixedPadding(filters=64, kernel_size=3, strides=2, name=name + '_stem_conv_4')(x)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, name=name + '_stem_batch_norm_4')(x)
        x = layers.Activation(activation, name=name + '_stem_act_4')(x)
        return x
    return apply