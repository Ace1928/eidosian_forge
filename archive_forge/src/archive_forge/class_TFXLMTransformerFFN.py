from __future__ import annotations
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlm import XLMConfig
class TFXLMTransformerFFN(keras.layers.Layer):

    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs):
        super().__init__(**kwargs)
        self.lin1 = keras.layers.Dense(dim_hidden, kernel_initializer=get_initializer(config.init_std), name='lin1')
        self.lin2 = keras.layers.Dense(out_dim, kernel_initializer=get_initializer(config.init_std), name='lin2')
        self.act = get_tf_activation('gelu') if config.gelu_activation else get_tf_activation('relu')
        self.dropout = keras.layers.Dropout(config.dropout)
        self.in_dim = in_dim
        self.dim_hidden = dim_hidden

    def call(self, input, training=False):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'lin1', None) is not None:
            with tf.name_scope(self.lin1.name):
                self.lin1.build([None, None, self.in_dim])
        if getattr(self, 'lin2', None) is not None:
            with tf.name_scope(self.lin2.name):
                self.lin2.build([None, None, self.dim_hidden])