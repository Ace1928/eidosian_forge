from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_gptj import GPTJConfig
class TFGPTJBlock(keras.layers.Layer):

    def __init__(self, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_1')
        self.attn = TFGPTJAttention(config, name='attn')
        self.mlp = TFGPTJMLP(inner_dim, config, name='mlp')
        self.config = config

    def call(self, hidden_states: tf.Tensor, layer_past: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, use_cache: bool=False, output_attentions: bool=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states=hidden_states, layer_past=layer_past, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'ln_1', None) is not None:
            with tf.name_scope(self.ln_1.name):
                self.ln_1.build([None, None, self.config.n_embd])
        if getattr(self, 'attn', None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        if getattr(self, 'mlp', None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)