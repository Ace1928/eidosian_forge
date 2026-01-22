import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class TFEfficientFormerDenseMlp(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear_in = keras.layers.Dense(units=hidden_features, kernel_initializer=get_initializer(config.initializer_range), name='linear_in')
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.linear_out = keras.layers.Dense(units=out_features, kernel_initializer=get_initializer(config.initializer_range), name='linear_out')
        self.hidden_features = hidden_features
        self.in_features = in_features

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_states = self.linear_in(inputs=hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.linear_out(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'linear_in', None) is not None:
            with tf.name_scope(self.linear_in.name):
                self.linear_in.build([None, None, self.in_features])
        if getattr(self, 'linear_out', None) is not None:
            with tf.name_scope(self.linear_out.name):
                self.linear_out.build([None, None, self.hidden_features])