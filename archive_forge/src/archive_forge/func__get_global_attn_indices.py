from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_longformer import LongformerConfig
@staticmethod
def _get_global_attn_indices(is_index_global_attn):
    """compute global attn indices required throughout forward pass"""
    num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
    num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)
    max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)
    is_index_global_attn_nonzero = tf.where(is_index_global_attn)
    is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(num_global_attn_indices, axis=-1)
    is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)
    is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(is_local_index_global_attn))
    return (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero)