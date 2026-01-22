from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_opt import OPTConfig
class TFOPTLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def call(self, attention_mask, past_key_values_length: int=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = tf.cast(attention_mask, tf.int64)
        positions = tf.math.cumsum(attention_mask, axis=1) * attention_mask - 1
        positions = positions[:, past_key_values_length:]
        return super().call(positions + self.offset)