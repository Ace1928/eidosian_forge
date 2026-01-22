from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
class TFEsmAttention(keras.layers.Layer):

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.self = TFEsmSelfAttention(config, name='self')
        self.output_layer = TFEsmSelfOutput(config, name='output')
        self.pruned_heads = set()
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.config = config

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, training=False):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(hidden_states_ln, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, training)
        attention_output = self.output_layer(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self', None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, 'output_layer', None) is not None:
            with tf.name_scope(self.output_layer.name):
                self.output_layer.build(None)
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])