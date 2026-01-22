from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
class TFSegformerMixFFN(keras.layers.Layer):

    def __init__(self, config: SegformerConfig, in_features: int, hidden_features: int=None, out_features: int=None, **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        self.dense1 = keras.layers.Dense(hidden_features, name='dense1')
        self.depthwise_convolution = TFSegformerDWConv(hidden_features, name='dwconv')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = keras.layers.Dense(out_features, name='dense2')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.hidden_features = hidden_features
        self.in_features = in_features

    def call(self, hidden_states: tf.Tensor, height: int, width: int, training: bool=False) -> tf.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.depthwise_convolution(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense1', None) is not None:
            with tf.name_scope(self.dense1.name):
                self.dense1.build([None, None, self.in_features])
        if getattr(self, 'depthwise_convolution', None) is not None:
            with tf.name_scope(self.depthwise_convolution.name):
                self.depthwise_convolution.build(None)
        if getattr(self, 'dense2', None) is not None:
            with tf.name_scope(self.dense2.name):
                self.dense2.build([None, None, self.hidden_features])