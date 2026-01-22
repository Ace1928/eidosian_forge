from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ctrl import CTRLConfig
class TFMultiHeadAttention(keras.layers.Layer):

    def __init__(self, d_model_size, num_heads, output_attentions=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.output_attentions = output_attentions
        self.depth = int(d_model_size / self.num_heads)
        self.Wq = keras.layers.Dense(d_model_size, name='Wq')
        self.Wk = keras.layers.Dense(d_model_size, name='Wk')
        self.Wv = keras.layers.Dense(d_model_size, name='Wv')
        self.dense = keras.layers.Dense(d_model_size, name='dense')

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        batch_size = shape_list(q)[0]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            k = tf.concat((past_key, k), axis=-2)
            v = tf.concat((past_value, v), axis=-2)
        if use_cache:
            present = tf.stack((k, v), axis=0)
        else:
            present = (None,)
        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = tf.transpose(output[0], perm=[0, 2, 1, 3])
        attn = output[1]
        original_size_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model_size))
        output = self.dense(original_size_attention)
        outputs = (output, present)
        if output_attentions:
            outputs = outputs + (attn,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'Wq', None) is not None:
            with tf.name_scope(self.Wq.name):
                self.Wq.build([None, None, self.d_model_size])
        if getattr(self, 'Wk', None) is not None:
            with tf.name_scope(self.Wk.name):
                self.Wk.build([None, None, self.d_model_size])
        if getattr(self, 'Wv', None) is not None:
            with tf.name_scope(self.Wv.name):
                self.Wv.build([None, None, self.d_model_size])
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.d_model_size])