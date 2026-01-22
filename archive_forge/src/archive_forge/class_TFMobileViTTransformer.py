from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
class TFMobileViTTransformer(keras.layers.Layer):

    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = []
        for i in range(num_stages):
            transformer_layer = TFMobileViTTransformerLayer(config, hidden_size=hidden_size, intermediate_size=int(hidden_size * config.mlp_ratio), name=f'layer.{i}')
            self.layers.append(transformer_layer)

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layers', None) is not None:
            for layer_module in self.layers:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)