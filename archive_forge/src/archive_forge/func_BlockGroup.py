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
def BlockGroup(filters, strides, num_repeats, se_ratio: float=0.25, bn_epsilon: float=1e-05, bn_momentum: float=0.0, activation: str='relu', survival_probability: float=0.8, name=None):
    """Create one group of blocks for the ResNet model."""
    if name is None:
        counter = backend.get_uid('block_group_')
        name = f'block_group_{counter}'

    def apply(inputs):
        x = BottleneckBlock(filters=filters, strides=strides, use_projection=True, se_ratio=se_ratio, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, activation=activation, survival_probability=survival_probability, name=name + '_block_0_')(inputs)
        for i in range(1, num_repeats):
            x = BottleneckBlock(filters=filters, strides=1, use_projection=False, se_ratio=se_ratio, activation=activation, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, survival_probability=survival_probability, name=name + f'_block_{i}_')(x)
        return x
    return apply