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
class _OrderedDictNumpyFeedFn(object):
    """Creates feed dictionaries from `OrderedDict`s of numpy arrays."""

    def __init__(self, placeholders, ordered_dict_of_arrays, batch_size, random_start=False, seed=None, num_epochs=None):
        if len(placeholders) != len(ordered_dict_of_arrays) + 1:
            raise ValueError('Expected {} placeholders; got {}.'.format(len(ordered_dict_of_arrays), len(placeholders)))
        self._index_placeholder = placeholders[0]
        self._col_placeholders = placeholders[1:]
        self._ordered_dict_of_arrays = ordered_dict_of_arrays
        self._max = len(next(iter(ordered_dict_of_arrays.values())))
        for _, v in ordered_dict_of_arrays.items():
            if len(v) != self._max:
                raise ValueError('Array lengths must match.')
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._epoch = 0
        random.seed(seed)
        self._trav = random.randrange(self._max) if random_start else 0
        self._epoch_end = (self._trav - 1) % self._max

    def __call__(self):
        integer_indexes, self._epoch = _get_integer_indices_for_next_batch(batch_indices_start=self._trav, batch_size=self._batch_size, epoch_end=self._epoch_end, array_length=self._max, current_epoch=self._epoch, total_epochs=self._num_epochs)
        self._trav = (integer_indexes[-1] + 1) % self._max
        feed_dict = {self._index_placeholder: integer_indexes}
        cols = [column[integer_indexes] for column in self._ordered_dict_of_arrays.values()]
        feed_dict.update(dict(zip(self._col_placeholders, cols)))
        return feed_dict