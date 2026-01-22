from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class TFGroupViTAttention(keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.embed_dim // self.num_attention_heads
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_attention_heads}).')
        factor = config.initializer_factor
        in_proj_std = self.embed_dim ** (-0.5) * (2 * config.num_hidden_layers) ** (-0.5) * factor
        out_proj_std = self.embed_dim ** (-0.5) * factor
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.q_proj = keras.layers.Dense(units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name='q_proj')
        self.k_proj = keras.layers.Dense(units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name='k_proj')
        self.v_proj = keras.layers.Dense(units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name='v_proj')
        self.dropout = keras.layers.Dropout(rate=config.attention_dropout)
        self.out_proj = keras.layers.Dense(units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name='out_proj')

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor=None, causal_attention_mask: tf.Tensor=None, output_attentions: bool=None, encoder_hidden_states: tf.Tensor=None, training: bool=False) -> Tuple[tf.Tensor]:
        """Input shape: Batch x Time x Channel"""
        batch_size = shape_list(hidden_states)[0]
        is_cross_attention = encoder_hidden_states is not None
        mixed_query_layer = self.q_proj(inputs=hidden_states)
        if is_cross_attention:
            mixed_key_layer = self.k_proj(inputs=encoder_hidden_states)
            mixed_value_layer = self.v_proj(inputs=encoder_hidden_states)
        else:
            mixed_key_layer = self.k_proj(inputs=hidden_states)
            mixed_value_layer = self.v_proj(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)
        if causal_attention_mask is not None:
            attention_scores = tf.add(attention_scores, causal_attention_mask)
        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask)
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.dropout(inputs=_attention_probs)
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))
        attention_output = self.out_proj(attention_output)
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'q_proj', None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, 'k_proj', None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, 'v_proj', None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, 'out_proj', None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])