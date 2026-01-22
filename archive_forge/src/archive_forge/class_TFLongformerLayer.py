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
class TFLongformerLayer(keras.layers.Layer):

    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFLongformerAttention(config, layer_id, name='attention')
        self.intermediate = TFLongformerIntermediate(config, name='intermediate')
        self.longformer_output = TFLongformerOutput(config, name='output')

    def call(self, inputs, training=False):
        hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn = inputs
        attention_outputs = self.attention([hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn], training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.longformer_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'intermediate', None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, 'longformer_output', None) is not None:
            with tf.name_scope(self.longformer_output.name):
                self.longformer_output.build(None)