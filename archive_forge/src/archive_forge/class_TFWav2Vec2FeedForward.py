from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
class TFWav2Vec2FeedForward(keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_dropout = keras.layers.Dropout(config.activation_dropout)
        self.intermediate_dense = keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='intermediate_dense')
        self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        self.output_dense = keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='output_dense')
        self.output_dropout = keras.layers.Dropout(config.hidden_dropout)
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, training=training)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'intermediate_dense', None) is not None:
            with tf.name_scope(self.intermediate_dense.name):
                self.intermediate_dense.build([None, None, self.config.hidden_size])
        if getattr(self, 'output_dense', None) is not None:
            with tf.name_scope(self.output_dense.name):
                self.output_dense.build([None, None, self.config.intermediate_size])