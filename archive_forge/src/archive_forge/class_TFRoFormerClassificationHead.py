from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_roformer import RoFormerConfig
class TFRoFormerClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.dense = keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.out_proj = keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='out_proj')
        if isinstance(config.hidden_act, str):
            self.classifier_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.classifier_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.classifier_act_fn(hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, 'out_proj', None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])