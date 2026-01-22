from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
class TFSegformerMLP(keras.layers.Layer):
    """
    Linear Embedding.
    """

    def __init__(self, input_dim: int, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.proj = keras.layers.Dense(config.decoder_hidden_size, name='proj')
        self.input_dim = input_dim

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        height = shape_list(hidden_states)[1]
        width = shape_list(hidden_states)[2]
        hidden_dim = shape_list(hidden_states)[-1]
        hidden_states = tf.reshape(hidden_states, (-1, height * width, hidden_dim))
        hidden_states = self.proj(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'proj', None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, self.input_dim])