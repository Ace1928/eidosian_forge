from __future__ import annotations
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig
def torch_gather(x, indices, gather_axis):
    if gather_axis < 0:
        gather_axis = tf.rank(x) + gather_axis
    if gather_axis != tf.rank(x) - 1:
        pre_roll = tf.rank(x) - 1 - gather_axis
        permutation = tf.roll(tf.range(tf.rank(x)), pre_roll, axis=0)
        x = tf.transpose(x, perm=permutation)
        indices = tf.transpose(indices, perm=permutation)
    else:
        pre_roll = 0
    flat_x = tf.reshape(x, (-1, tf.shape(x)[-1]))
    flat_indices = tf.reshape(indices, (-1, tf.shape(indices)[-1]))
    gathered = tf.gather(flat_x, flat_indices, batch_dims=1)
    gathered = tf.reshape(gathered, tf.shape(indices))
    if pre_roll != 0:
        permutation = tf.roll(tf.range(tf.rank(x)), -pre_roll, axis=0)
        gathered = tf.transpose(gathered, perm=permutation)
    return gathered