from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
import types as tp
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.inputs.queues import feeding_queue_runner as fqr
class _ArrayFeedFn(object):
    """Creates feed dictionaries from numpy arrays."""

    def __init__(self, placeholders, array, batch_size, random_start=False, seed=None, num_epochs=None):
        if len(placeholders) != 2:
            raise ValueError('_array_feed_fn expects 2 placeholders; got {}.'.format(len(placeholders)))
        self._placeholders = placeholders
        self._array = array
        self._max = len(array)
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._epoch = 0
        random.seed(seed)
        self._trav = random.randrange(self._max) if random_start else 0
        self._epoch_end = (self._trav - 1) % self._max

    def __call__(self):
        integer_indexes, self._epoch = _get_integer_indices_for_next_batch(batch_indices_start=self._trav, batch_size=self._batch_size, epoch_end=self._epoch_end, array_length=self._max, current_epoch=self._epoch, total_epochs=self._num_epochs)
        self._trav = (integer_indexes[-1] + 1) % self._max
        return {self._placeholders[0]: integer_indexes, self._placeholders[1]: self._array[integer_indexes]}