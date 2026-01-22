from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_convbert import ConvBertConfig
class TFConvBertSelfAttention(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        new_num_attention_heads = int(config.num_attention_heads / config.head_ratio)
        if new_num_attention_heads < 1:
            self.head_ratio = config.num_attention_heads
            num_attention_heads = 1
        else:
            num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio
        self.num_attention_heads = num_attention_heads
        self.conv_kernel_size = config.conv_kernel_size
        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError('hidden_size should be divisible by num_attention_heads')
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.key_conv_attn_layer = keras.layers.SeparableConv1D(self.all_head_size, self.conv_kernel_size, padding='same', activation=None, depthwise_initializer=get_initializer(1 / self.conv_kernel_size), pointwise_initializer=get_initializer(config.initializer_range), name='key_conv_attn_layer')
        self.conv_kernel_layer = keras.layers.Dense(self.num_attention_heads * self.conv_kernel_size, activation=None, name='conv_kernel_layer', kernel_initializer=get_initializer(config.initializer_range))
        self.conv_out_layer = keras.layers.Dense(self.all_head_size, activation=None, name='conv_out_layer', kernel_initializer=get_initializer(config.initializer_range))
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        conv_attn_layer = tf.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = tf.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        conv_kernel_layer = stable_softmax(conv_kernel_layer, axis=1)
        paddings = tf.constant([[0, 0], [int((self.conv_kernel_size - 1) / 2), int((self.conv_kernel_size - 1) / 2)], [0, 0]])
        conv_out_layer = self.conv_out_layer(hidden_states)
        conv_out_layer = tf.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        conv_out_layer = tf.pad(conv_out_layer, paddings, 'CONSTANT')
        unfold_conv_out_layer = tf.stack([tf.slice(conv_out_layer, [0, i, 0], [batch_size, shape_list(mixed_query_layer)[1], self.all_head_size]) for i in range(self.conv_kernel_size)], axis=-1)
        conv_out_layer = tf.reshape(unfold_conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        conv_out_layer = tf.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = tf.reshape(conv_out_layer, [-1, self.all_head_size])
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(shape_list(key_layer)[-1], attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        value_layer = tf.reshape(mixed_value_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        conv_out = tf.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        context_layer = tf.concat([context_layer, conv_out], 2)
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.head_ratio * self.all_head_size))
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'query', None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, 'key', None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, 'value', None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        if getattr(self, 'key_conv_attn_layer', None) is not None:
            with tf.name_scope(self.key_conv_attn_layer.name):
                self.key_conv_attn_layer.build([None, None, self.config.hidden_size])
        if getattr(self, 'conv_kernel_layer', None) is not None:
            with tf.name_scope(self.conv_kernel_layer.name):
                self.conv_kernel_layer.build([None, None, self.all_head_size])
        if getattr(self, 'conv_out_layer', None) is not None:
            with tf.name_scope(self.conv_out_layer.name):
                self.conv_out_layer.build([None, None, self.config.hidden_size])