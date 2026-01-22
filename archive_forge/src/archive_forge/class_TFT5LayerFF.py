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
class TFT5LayerFF(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.is_gated_act:
            self.DenseReluDense = TFT5DenseGatedActDense(config, name='DenseReluDense')
        else:
            self.DenseReluDense = TFT5DenseActDense(config, name='DenseReluDense')
        self.layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name='layer_norm')
        self.dropout = keras.layers.Dropout(config.dropout_rate)

    def call(self, hidden_states, training=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        dense_output = self.DenseReluDense(normed_hidden_states, training=training)
        hidden_states = hidden_states + self.dropout(dense_output, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build(None)
        if getattr(self, 'DenseReluDense', None) is not None:
            with tf.name_scope(self.DenseReluDense.name):
                self.DenseReluDense.build(None)