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
def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
    """pads rows and then flips rows and columns"""
    hidden_states_padded = tf.pad(hidden_states_padded, paddings)
    batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
    hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))
    return hidden_states_padded