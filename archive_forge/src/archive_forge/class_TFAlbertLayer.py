from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_albert import AlbertConfig
class TFAlbertLayer(keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFAlbertAttention(config, name='attention')
        self.ffn = keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='ffn')
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        self.ffn_output = keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='ffn_output')
        self.full_layer_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='full_layer_layer_norm')
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(input_tensor=hidden_states, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, training=training)
        ffn_output = self.ffn(inputs=attention_outputs[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(inputs=ffn_output)
        ffn_output = self.dropout(inputs=ffn_output, training=training)
        hidden_states = self.full_layer_layer_norm(inputs=ffn_output + attention_outputs[0])
        outputs = (hidden_states,) + attention_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'ffn', None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build([None, None, self.config.hidden_size])
        if getattr(self, 'ffn_output', None) is not None:
            with tf.name_scope(self.ffn_output.name):
                self.ffn_output.build([None, None, self.config.intermediate_size])
        if getattr(self, 'full_layer_layer_norm', None) is not None:
            with tf.name_scope(self.full_layer_layer_norm.name):
                self.full_layer_layer_norm.build([None, None, self.config.hidden_size])