import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class TFEfficientFormerPooling(keras.layers.Layer):

    def __init__(self, pool_size: int, **kwargs):
        super().__init__(**kwargs)
        self.pool = keras.layers.AveragePooling2D(pool_size=pool_size, strides=1, padding='same')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        output = self.pool(hidden_states)
        output = output - hidden_states
        return output