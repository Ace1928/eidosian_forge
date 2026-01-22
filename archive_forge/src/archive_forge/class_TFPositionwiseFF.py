from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ....modeling_tf_utils import (
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
class TFPositionwiseFF(keras.layers.Layer):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-05, init_std=0.02, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.layer_1 = keras.layers.Dense(d_inner, kernel_initializer=get_initializer(init_std), activation=tf.nn.relu, name='CoreNet_._0')
        self.drop_1 = keras.layers.Dropout(dropout)
        self.layer_2 = keras.layers.Dense(d_model, kernel_initializer=get_initializer(init_std), name='CoreNet_._3')
        self.drop_2 = keras.layers.Dropout(dropout)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='layer_norm')
        self.pre_lnorm = pre_lnorm

    def call(self, inp, training=False):
        if self.pre_lnorm:
            core_out = self.layer_norm(inp)
            core_out = self.layer_1(core_out)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)
            output = core_out + inp
        else:
            core_out = self.layer_1(inp)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)
            output = self.layer_norm(inp + core_out)
        return output