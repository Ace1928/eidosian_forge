from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
class TFLxmertSelfAttentionLayer(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self = TFLxmertAttention(config, name='self')
        self.attention_output = TFLxmertAttentionOutput(config, name='output')

    def call(self, input_tensor, attention_mask, output_attentions, training=False):
        self_output = self.self(input_tensor, input_tensor, attention_mask, output_attentions)
        if output_attentions:
            attention_probs = self_output[1]
        attention_output = self.attention_output(self_output[0], input_tensor)
        return (attention_output, attention_probs) if output_attentions else (attention_output,)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self', None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, 'attention_output', None) is not None:
            with tf.name_scope(self.attention_output.name):
                self.attention_output.build(None)