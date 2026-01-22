from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_opt import OPTConfig
class TFOPTDecoderLayer(keras.layers.Layer):

    def __init__(self, config: OPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.do_layer_norm_before = config.do_layer_norm_before
        self.embed_dim = config.hidden_size
        self.self_attn = TFOPTAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=config.attention_dropout, name='self_attn', is_decoder=True)
        self.dropout = keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='self_attn_layer_norm')
        self.fc1 = keras.layers.Dense(config.ffn_dim, name='fc1')
        self.fc2 = keras.layers.Dense(self.embed_dim, name='fc2')
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='final_layer_norm')
        self.config = config

    def call(self, hidden_states: tf.Tensor, attention_mask: np.ndarray | tf.Tensor | None=None, layer_head_mask: tf.Tensor | None=None, past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, training: Optional[bool]=False, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=False) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`, *optional*): mask for attention heads in a given layer of size
                `(decoder_attention_heads,)`
            past_key_value (`Tuple(tf.Tensor)`, *optional*): cached past key and value projection states
            training (`bool`, *optional*, defaults to `False`):
                Whether or not to use the model in training mode (some modules like dropout modules have different
                behaviors between training and evaluation).
        """
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return (hidden_states, self_attn_weights, present_key_value)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self_attn', None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        if getattr(self, 'self_attn_layer_norm', None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        if getattr(self, 'fc1', None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        if getattr(self, 'fc2', None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.ffn_dim])
        if getattr(self, 'final_layer_norm', None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])