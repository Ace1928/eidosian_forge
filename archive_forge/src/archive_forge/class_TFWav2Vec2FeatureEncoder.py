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
class TFWav2Vec2FeatureEncoder(keras.layers.Layer):

    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if config.feat_extract_norm == 'group':
            conv_layers = [TFWav2Vec2GroupNormConvLayer(config, layer_id=0, name=f'conv_layers.{0}')] + [TFWav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, name=f'conv_layers.{i + 1}') for i in range(config.num_feat_extract_layers - 1)]
        elif config.feat_extract_norm == 'layer':
            conv_layers = [TFWav2Vec2LayerNormConvLayer(config, layer_id=i, name=f'conv_layers.{i}') for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']")
        self.conv_layers = conv_layers

    def call(self, input_values):
        hidden_states = tf.expand_dims(input_values, -1)
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv_layers', None) is not None:
            for conv_layer in self.conv_layers:
                with tf.name_scope(conv_layer.name):
                    conv_layer.build(None)