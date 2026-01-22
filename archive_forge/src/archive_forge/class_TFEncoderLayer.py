from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ctrl import CTRLConfig
class TFEncoderLayer(keras.layers.Layer):

    def __init__(self, d_model_size, num_heads, dff, rate=0.1, layer_norm_epsilon=1e-06, output_attentions=False, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = output_attentions
        self.multi_head_attention = TFMultiHeadAttention(d_model_size, num_heads, output_attentions=self.output_attentions, name='multi_head_attention')
        self.ffn = TFPointWiseFeedForwardLayer(d_model_size, dff, name='ffn')
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='layernorm1')
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='layernorm2')
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.d_model_size = d_model_size

    def call(self, x, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(normed, normed, normed, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=training)
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output
        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output
        outputs = (out2,) + attn_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'multi_head_attention', None) is not None:
            with tf.name_scope(self.multi_head_attention.name):
                self.multi_head_attention.build(None)
        if getattr(self, 'ffn', None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)
        if getattr(self, 'layernorm1', None) is not None:
            with tf.name_scope(self.layernorm1.name):
                self.layernorm1.build([None, None, self.d_model_size])
        if getattr(self, 'layernorm2', None) is not None:
            with tf.name_scope(self.layernorm2.name):
                self.layernorm2.build([None, None, self.d_model_size])