from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel
class TFBlipAttention(keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** (-0.5)
        self.dropout = keras.layers.Dropout(config.attention_dropout, name='dropout')
        self.qkv = keras.layers.Dense(3 * self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='qkv')
        self.projection = keras.layers.Dense(self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='projection')

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: Optional[bool]=None) -> Tuple[tf.Tensor, tf.Tensor | None, Tuple[tf.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = shape_list(hidden_states)
        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = tf.reshape(mixed_qkv, (bsz, tgt_len, 3, self.num_heads, self.head_dim))
        mixed_qkv = tf.transpose(mixed_qkv, perm=(2, 0, 3, 1, 4))
        query_states, key_states, value_states = (mixed_qkv[0], mixed_qkv[1], mixed_qkv[2])
        attention_scores = query_states @ tf.transpose(key_states, (0, 1, 3, 2))
        attention_scores = attention_scores * self.scale
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = tf.transpose(attention_probs @ value_states, perm=(0, 2, 1, 3))
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.embed_dim]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        output = self.projection(context_layer)
        outputs = (output, attention_probs) if output_attentions else (output, None)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, 'qkv', None) is not None:
            with tf.name_scope(self.qkv.name):
                self.qkv.build([None, None, self.embed_dim])
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, self.embed_dim])