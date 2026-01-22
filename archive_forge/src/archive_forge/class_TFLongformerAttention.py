from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_longformer import LongformerConfig
class TFLongformerAttention(keras.layers.Layer):

    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFLongformerSelfAttention(config, layer_id, name='self')
        self.dense_output = TFLongformerSelfOutput(config, name='output')

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn = inputs
        self_outputs = self.self_attention([hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn], training=training)
        attention_output = self.dense_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self_attention', None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        if getattr(self, 'dense_output', None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)