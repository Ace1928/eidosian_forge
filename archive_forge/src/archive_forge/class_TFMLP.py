from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_openai import OpenAIGPTConfig
class TFMLP(keras.layers.Layer):

    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name='c_fc')
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name='c_proj')
        self.act = get_tf_activation('gelu')
        self.dropout = keras.layers.Dropout(config.resid_pdrop)
        self.nx = nx
        self.n_state = n_state

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'c_fc', None) is not None:
            with tf.name_scope(self.c_fc.name):
                self.c_fc.build([None, None, self.n_state])
        if getattr(self, 'c_proj', None) is not None:
            with tf.name_scope(self.c_proj.name):
                self.c_proj.build([None, None, self.nx])