from __future__ import annotations
import copy
import itertools
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_t5 import T5Config
class TFT5LayerCrossAttention(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.EncDecAttention = TFT5Attention(config, has_relative_attention_bias=False, name='EncDecAttention')
        self.layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name='layer_norm')
        self.dropout = keras.layers.Dropout(config.dropout_rate)

    def call(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, query_length=None, use_cache=False, output_attentions=False, training=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(normed_hidden_states, mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, query_length=query_length, use_cache=use_cache, output_attentions=output_attentions, training=training)
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        outputs = (hidden_states,) + attention_output[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'EncDecAttention', None) is not None:
            with tf.name_scope(self.EncDecAttention.name):
                self.EncDecAttention.build(None)
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build(None)