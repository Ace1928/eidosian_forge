from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_bart import BartConfig
class TFBartLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def call(self, input_shape: Optional[tf.TensorShape]=None, past_key_values_length: int=0, position_ids: tf.Tensor | None=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name='range')
            position_ids += past_key_values_length
        offset_dtype = position_ids.dtype if isinstance(position_ids, tf.Tensor) else tf.int32
        return super().call(position_ids + tf.constant(self.offset, dtype=offset_dtype))