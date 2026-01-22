from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class TFSamVisionAttention(keras.layers.Layer):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size, **kwargs):
        super().__init__(**kwargs)
        input_size = (config.image_size // config.patch_size, config.image_size // config.patch_size) if window_size == 0 else (window_size, window_size)
        self.input_size = input_size
        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        self.scale = head_dim ** (-0.5)
        self.dropout = config.attention_dropout
        self.qkv = keras.layers.Dense(config.hidden_size * 3, use_bias=config.qkv_bias, name='qkv')
        self.proj = keras.layers.Dense(config.hidden_size, name='proj')
        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError('Input size must be provided if using relative positional encoding.')
        self.config = config

    def build(self, input_shape=None):
        if self.input_size is not None:
            self.rel_pos_h = self.add_weight(shape=(2 * self.input_size[0] - 1, self.head_dim), initializer='zeros', name='rel_pos_h')
            self.rel_pos_w = self.add_weight(shape=(2 * self.input_size[1] - 1, self.head_dim), initializer='zeros', name='rel_pos_w')
        if self.built:
            return
        self.built = True
        if getattr(self, 'qkv', None) is not None:
            with tf.name_scope(self.qkv.name):
                self.qkv.build([None, None, self.config.hidden_size])
        if getattr(self, 'proj', None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, self.config.hidden_size])

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: tf.Tensor) -> tf.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`tf.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos_resized = tf.image.resize(tf.reshape(rel_pos, (1, rel_pos.shape[0], -1)), size=(max_rel_dist, rel_pos.shape[1]), method='bilinear')
            rel_pos_resized = tf.reshape(rel_pos_resized, (-1, max_rel_dist))
        else:
            rel_pos_resized = rel_pos
        q_coords = tf.expand_dims(tf.range(q_size, dtype=tf.float32), 1) * max(k_size / q_size, 1.0)
        k_coords = tf.expand_dims(tf.range(k_size, dtype=tf.float32), 0) * max(q_size / k_size, 1.0)
        relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
        return tf.gather(rel_pos_resized, tf.cast(relative_coords, tf.int32))

    def add_decomposed_rel_pos(self, attn: tf.Tensor, query: tf.Tensor, rel_pos_h: tf.Tensor, rel_pos_w: tf.Tensor, q_size: Tuple[int, int], k_size: Tuple[int, int]) -> tf.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`tf.Tensor`):
                attention map.
            query (`tf.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`tf.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`tf.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`tf.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)
        batch_size, _, dim = shape_list(query)
        reshaped_query = tf.reshape(query, (batch_size, query_height, query_width, dim))
        rel_h = tf.einsum('bhwc,hkc->bhwk', reshaped_query, relative_position_height)
        rel_w = tf.einsum('bhwc,wkc->bhwk', reshaped_query, relative_position_width)
        attn = tf.reshape(attn, (batch_size, query_height, query_width, key_height, key_width))
        attn = attn + tf.expand_dims(rel_h, axis=-1) + tf.expand_dims(rel_w, axis=-2)
        attn = tf.reshape(attn, (batch_size, query_height * query_width, key_height * key_width))
        return attn

    def call(self, hidden_states: tf.Tensor, output_attentions=False, training=False) -> tf.Tensor:
        batch_size, height, width, _ = shape_list(hidden_states)
        qkv = tf.reshape(self.qkv(hidden_states), (batch_size, height * width, 3, self.num_attention_heads, -1))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        query, key, value = tf.unstack(tf.reshape(qkv, (3, batch_size * self.num_attention_heads, height * width, -1)), axis=0)
        attn_weights = tf.matmul(query * self.scale, key, transpose_b=True)
        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        if training:
            attn_probs = tf.nn.dropout(attn_weights, rate=self.dropout)
        else:
            attn_probs = attn_weights
        attn_output = tf.reshape(attn_probs @ value, (batch_size, self.num_attention_heads, height, width, -1))
        attn_output = tf.transpose(attn_output, perm=(0, 2, 3, 1, 4))
        attn_output = tf.reshape(attn_output, (batch_size, height, width, self.config.hidden_size))
        attn_output = self.proj(attn_output)
        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        return outputs