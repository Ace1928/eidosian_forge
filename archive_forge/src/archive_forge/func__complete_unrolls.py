import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
def _complete_unrolls(self, env_ids):
    env_indices = self._index.sparse_read(env_ids)
    env_ids = tf.gather(env_ids, tf.where(tf.equal(env_indices, self._full_length))[:, 0])
    env_ids = tf.cast(env_ids, tf.int64)
    unrolls = tf.nest.map_structure(lambda s: s.sparse_read(env_ids), self._state)
    j = self._num_overlapping_steps + 1
    repeated_start_range = tf.tile(tf.range(j, dtype=tf.int64), [tf.shape(env_ids)[0]])
    repeated_end_range = tf.tile(tf.range(self._full_length - j, self._full_length, dtype=tf.int64), [tf.shape(env_ids)[0]])
    repeated_env_ids = tf.reshape(tf.tile(tf.expand_dims(env_ids, -1), [1, j]), [-1])
    start_indices = tf.stack([repeated_env_ids, repeated_start_range], -1)
    end_indices = tf.stack([repeated_env_ids, repeated_end_range], -1)
    for s in tf.nest.flatten(self._state):
        s.scatter_nd_update(start_indices, s.gather_nd(end_indices))
    self._index.scatter_update(tf.IndexedSlices(1 + self._num_overlapping_steps, env_ids))
    return (env_ids, unrolls)