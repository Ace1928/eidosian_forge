from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_funnel import FunnelConfig
class TFFunnelClassificationHead(keras.layers.Layer):

    def __init__(self, config, n_labels, **kwargs):
        super().__init__(**kwargs)
        initializer = get_initializer(config.initializer_range)
        self.linear_hidden = keras.layers.Dense(config.d_model, kernel_initializer=initializer, name='linear_hidden')
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        self.linear_out = keras.layers.Dense(n_labels, kernel_initializer=initializer, name='linear_out')
        self.config = config

    def call(self, hidden, training=False):
        hidden = self.linear_hidden(hidden)
        hidden = keras.activations.tanh(hidden)
        hidden = self.dropout(hidden, training=training)
        return self.linear_out(hidden)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'linear_hidden', None) is not None:
            with tf.name_scope(self.linear_hidden.name):
                self.linear_hidden.build([None, None, self.config.d_model])
        if getattr(self, 'linear_out', None) is not None:
            with tf.name_scope(self.linear_out.name):
                self.linear_out.build([None, None, self.config.d_model])