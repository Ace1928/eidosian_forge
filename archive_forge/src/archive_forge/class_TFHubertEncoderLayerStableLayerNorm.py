from __future__ import annotations
import warnings
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_hubert import HubertConfig
class TFHubertEncoderLayerStableLayerNorm(keras.layers.Layer):

    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFHubertAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=False, name='attention')
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.feed_forward = TFHubertFeedForward(config, name='feed_forward')
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='final_layer_norm')
        self.config = config

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: bool=False) -> Tuple[tf.Tensor]:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(hidden_states, attention_mask=attention_mask, training=training)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        if getattr(self, 'feed_forward', None) is not None:
            with tf.name_scope(self.feed_forward.name):
                self.feed_forward.build(None)
        if getattr(self, 'final_layer_norm', None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])