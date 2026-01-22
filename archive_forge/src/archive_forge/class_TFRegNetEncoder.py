from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
class TFRegNetEncoder(keras.layers.Layer):

    def __init__(self, config: RegNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.stages = []
        self.stages.append(TFRegNetStage(config, config.embedding_size, config.hidden_sizes[0], stride=2 if config.downsample_in_first_stage else 1, depth=config.depths[0], name='stages.0'))
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, config.depths[1:])):
            self.stages.append(TFRegNetStage(config, in_channels, out_channels, depth=depth, name=f'stages.{i + 1}'))

    def call(self, hidden_state: tf.Tensor, output_hidden_states: bool=False, return_dict: bool=True) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            hidden_state = stage_module(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, hidden_states] if v is not None))
        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)