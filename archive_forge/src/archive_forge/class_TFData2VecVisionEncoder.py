from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
class TFData2VecVisionEncoder(keras.layers.Layer):

    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple]=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(config, window_size=window_size, name='relative_position_bias')
        else:
            self.relative_position_bias = None
        dpr = list(tf.linspace(0.0, config.drop_path_rate, config.num_hidden_layers))
        self.layer = [TFData2VecVisionLayer(config, window_size=window_size if config.use_relative_position_bias else None, drop_path_rate=dpr[i], name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor | None=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            relative_position_bias = self.relative_position_bias(0.0) if self.relative_position_bias is not None else None
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'relative_position_bias', None) is not None:
            with tf.name_scope(self.relative_position_bias.name):
                self.relative_position_bias.build(None)
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)