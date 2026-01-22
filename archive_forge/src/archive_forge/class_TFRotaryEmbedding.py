from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
class TFRotaryEmbedding(keras.layers.Layer):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int, name=None):
        super().__init__(name=name)
        self.dim = dim

    def build(self, input_shape):
        super().build(input_shape)
        self.inv_freq = self.add_weight('inv_freq', shape=(self.dim // 2,), dtype=tf.float32, initializer=get_initializer(1.0), trainable=False)
        self.inv_freq.assign(1.0 / 10000 ** (tf.range(start=0, limit=self.dim, delta=2, dtype=tf.float32) / self.dim))

    def _compute_cos_sin(self, x, seq_dimension=2):
        seq_len = tf.shape(x)[seq_dimension]
        t = tf.range(seq_len, dtype=self.inv_freq.dtype)
        freqs = tf.einsum('i, j -> ij', t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)[None, None, :, :]
        return (tf.cos(emb), tf.sin(emb))

    def call(self, q: tf.Tensor, k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        cos_emb, sin_emb = self._compute_cos_sin(k, seq_dimension=-2)
        return (apply_rotary_pos_emb(q, cos_emb, sin_emb), apply_rotary_pos_emb(k, cos_emb, sin_emb))