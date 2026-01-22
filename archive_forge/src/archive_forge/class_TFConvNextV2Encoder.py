from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_convnextv2 import ConvNextV2Config
class TFConvNextV2Encoder(keras.layers.Layer):

    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        self.stages = []
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate, sum(config.depths))
        drop_path_rates = tf.split(drop_path_rates, config.depths)
        drop_path_rates = [x.numpy().tolist() for x in drop_path_rates]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = TFConvNextV2Stage(config, in_channels=prev_chs, out_channels=out_chs, stride=2 if i > 0 else 1, depth=config.depths[i], drop_path_rates=drop_path_rates[i], name=f'stages.{i}')
            self.stages.append(stage)
            prev_chs = out_chs

    def call(self, hidden_states: tf.Tensor, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, TFBaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    def build(self, input_shape=None):
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)