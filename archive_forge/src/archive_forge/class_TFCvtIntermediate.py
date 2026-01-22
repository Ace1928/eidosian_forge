from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
class TFCvtIntermediate(keras.layers.Layer):
    """Intermediate dense layer. Second chunk of the convolutional transformer block."""

    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(units=int(embed_dim * mlp_ratio), kernel_initializer=get_initializer(config.initializer_range), activation='gelu', name='dense')
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        hidden_state = self.dense(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.embed_dim])