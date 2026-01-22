from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
def _cal_pos_emb(self, dense_layer: keras.layers.Dense, position_ids: tf.Tensor, num_buckets: int, max_distance: int):
    rel_pos_matrix = tf.expand_dims(position_ids, axis=-2) - tf.expand_dims(position_ids, axis=-1)
    rel_pos = self.relative_position_bucket(rel_pos_matrix, num_buckets, max_distance)
    rel_pos_one_hot = tf.one_hot(rel_pos, depth=num_buckets, dtype=self.compute_dtype)
    embedding = dense_layer(rel_pos_one_hot)
    embedding = tf.transpose(embedding, [0, 3, 1, 2])
    embedding = tf.cast(embedding, dtype=self.compute_dtype)
    return embedding