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
from .configuration_pegasus import PegasusConfig
class TFPegasusSinusoidalPositionalEmbedding(keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f'odd embedding_dim {embedding_dim} not supported')
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions

    def build(self, input_shape: tf.TensorShape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        weight = self._init_weight(self.num_positions, self.embedding_dim)
        self.weight = self.add_weight(name='embeddings', shape=[self.num_positions, self.embedding_dim])
        weight = tf.cast(weight, dtype=self.weight.dtype)
        self.weight.assign(weight)
        super().build(input_shape)

    @staticmethod
    def _init_weight(n_pos: int, dim: int):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
        table = np.zeros_like(position_enc)
        table[:, 0:dim // 2] = np.sin(position_enc[:, 0::2])
        table[:, dim // 2:] = np.cos(position_enc[:, 1::2])
        table = tf.convert_to_tensor(table)
        tf.stop_gradient(table)
        return table

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int=0, position_ids: tf.Tensor | None=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(past_key_values_length, seq_len + past_key_values_length, delta=1, name='range')
        return tf.gather(self.weight, position_ids)