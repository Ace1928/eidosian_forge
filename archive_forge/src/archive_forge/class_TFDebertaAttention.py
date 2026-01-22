from __future__ import annotations
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig
class TFDebertaAttention(keras.layers.Layer):

    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        self.self = TFDebertaDisentangledSelfAttention(config, name='self')
        self.dense_output = TFDebertaSelfOutput(config, name='output')
        self.config = config

    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, query_states: tf.Tensor=None, relative_pos: tf.Tensor=None, rel_embeddings: tf.Tensor=None, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor]:
        self_outputs = self.self(hidden_states=input_tensor, attention_mask=attention_mask, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings, output_attentions=output_attentions, training=training)
        if query_states is None:
            query_states = input_tensor
        attention_output = self.dense_output(hidden_states=self_outputs[0], input_tensor=query_states, training=training)
        output = (attention_output,) + self_outputs[1:]
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self', None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, 'dense_output', None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)