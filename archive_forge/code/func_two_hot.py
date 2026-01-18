import logging
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.annotations import PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import (
@PublicAPI
def two_hot(value: 'tf.Tensor', num_buckets: int=255, lower_bound: float=-20.0, upper_bound: float=20.0, dtype=None):
    """Returns a two-hot vector of dim=num_buckets with two entries that are non-zero.

    See [1] for more details:
    [1] Mastering Diverse Domains through World Models - 2023
    D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
    https://arxiv.org/pdf/2301.04104v1.pdf

    Entries in the vector represent equally sized buckets within some fixed range
    (`lower_bound` to `upper_bound`).
    Those entries not 0.0 at positions k and k+1 encode the actual `value` and sum
    up to 1.0. They are the weights multiplied by the buckets values at k and k+1 for
    retrieving `value`.

    Example:
        num_buckets=11
        lower_bound=-5
        upper_bound=5
        value=2.5
        -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        -> [-5   -4   -3   -2   -1   0    1    2    3    4    5] (0.5*2 + 0.5*3=2.5)

    Example:
        num_buckets=5
        lower_bound=-1
        upper_bound=1
        value=0.1
        -> [0.0, 0.0, 0.8, 0.2, 0.0]
        -> [-1  -0.5   0   0.5   1] (0.2*0.5 + 0.8*0=0.1)

    Args:
        value: The input tensor of shape (B,) to be two-hot encoded.
        num_buckets: The number of buckets to two-hot encode into.
        lower_bound: The lower bound value used for the encoding. If input values are
            lower than this boundary, they will be encoded as `lower_bound`.
        upper_bound: The upper bound value used for the encoding. If input values are
            higher than this boundary, they will be encoded as `upper_bound`.

    Returns:
        The two-hot encoded tensor of shape (B, num_buckets).
    """
    value = tf.clip_by_value(value, lower_bound, upper_bound)
    batch_indices = tf.cast(tf.range(0, tf.shape(value)[0]), dtype=dtype or tf.float32)
    bucket_delta = (upper_bound - lower_bound) / (num_buckets - 1)
    idx = (-lower_bound + value) / bucket_delta
    k = tf.math.floor(idx)
    kp1 = tf.math.ceil(idx)
    kp1 = tf.where(tf.equal(k, kp1), kp1 + 1.0, kp1)
    kp1 = tf.where(tf.equal(kp1, num_buckets), kp1 - 2.0, kp1)
    values_k = lower_bound + k * bucket_delta
    values_kp1 = lower_bound + kp1 * bucket_delta
    weights_k = (value - values_kp1) / (values_k - values_kp1)
    weights_kp1 = 1.0 - weights_k
    indices_k = tf.stack([batch_indices, k], -1)
    indices_kp1 = tf.stack([batch_indices, kp1], -1)
    indices = tf.concat([indices_k, indices_kp1], 0)
    updates = tf.concat([weights_k, weights_kp1], 0)
    return tf.scatter_nd(tf.cast(indices, tf.int32), updates, shape=(tf.shape(value)[0], num_buckets))