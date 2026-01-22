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
class TFCvtAttention(keras.layers.Layer):
    """Attention layer. First chunk of the convolutional transformer block."""

    def __init__(self, config: CvtConfig, num_heads: int, embed_dim: int, kernel_size: int, stride_q: int, stride_kv: int, padding_q: int, padding_kv: int, qkv_projection_method: str, qkv_bias: bool, attention_drop_rate: float, drop_rate: float, with_cls_token: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFCvtSelfAttention(config, num_heads, embed_dim, kernel_size, stride_q, stride_kv, padding_q, padding_kv, qkv_projection_method, qkv_bias, attention_drop_rate, with_cls_token, name='attention')
        self.dense_output = TFCvtSelfOutput(config, embed_dim, drop_rate, name='output')

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool=False):
        self_output = self.attention(hidden_state, height, width, training=training)
        attention_output = self.dense_output(self_output, training=training)
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