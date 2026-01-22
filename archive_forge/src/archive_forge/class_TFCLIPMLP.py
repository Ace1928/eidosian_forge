from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
class TFCLIPMLP(keras.layers.Layer):

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)
        self.activation_fn = get_tf_activation(config.hidden_act)
        factor = config.initializer_factor
        in_proj_std = config.hidden_size ** (-0.5) * (2 * config.num_hidden_layers) ** (-0.5) * factor
        fc_std = (2 * config.hidden_size) ** (-0.5) * factor
        self.fc1 = keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name='fc1')
        self.fc2 = keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name='fc2')
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.fc1(inputs=hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(inputs=hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'fc1', None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.config.hidden_size])
        if getattr(self, 'fc2', None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.intermediate_size])