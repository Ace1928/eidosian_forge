from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class TFFFNLayer(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.intermediate = TFMobileBertIntermediate(config, name='intermediate')
        self.mobilebert_output = TFFFNOutput(config, name='output')

    def call(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.mobilebert_output(intermediate_output, hidden_states)
        return layer_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'intermediate', None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, 'mobilebert_output', None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)