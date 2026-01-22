from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
class TFRegNetStage(keras.layers.Layer):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int=2, depth: int=2, **kwargs):
        super().__init__(**kwargs)
        layer = TFRegNetXLayer if config.layer_type == 'x' else TFRegNetYLayer
        self.layers = [layer(config, in_channels, out_channels, stride=stride, name='layers.0'), *[layer(config, out_channels, out_channels, name=f'layers.{i + 1}') for i in range(depth - 1)]]

    def call(self, hidden_state):
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)