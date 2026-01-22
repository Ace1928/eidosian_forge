from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
class TFMobileViTAttention(keras.layers.Layer):

    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = TFMobileViTSelfAttention(config, hidden_size, name='attention')
        self.dense_output = TFMobileViTSelfOutput(config, hidden_size, name='output')

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        self_outputs = self.attention(hidden_states, training=training)
        attention_output = self.dense_output(self_outputs, training=training)
        return attention_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'dense_output', None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)