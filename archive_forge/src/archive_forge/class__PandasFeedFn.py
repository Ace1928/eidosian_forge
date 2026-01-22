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
class _PandasFeedFn(object):
    """Creates feed dictionaries from pandas `DataFrames`."""

    def __init__(self, placeholders, dataframe, batch_size, random_start=False, seed=None, num_epochs=None):
        if len(placeholders) != len(dataframe.columns) + 1:
            raise ValueError('Expected {} placeholders; got {}.'.format(len(dataframe.columns) + 1, len(placeholders)))
        self._index_placeholder = placeholders[0]
        self._col_placeholders = placeholders[1:]
        self._dataframe = dataframe
        self._max = len(dataframe)
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._epoch = 0
        random.seed(seed)
        self._trav = random.randrange(self._max) if random_start else 0
        self._epoch_end = (self._trav - 1) % self._max

    def __call__(self):
        integer_indexes, self._epoch = _get_integer_indices_for_next_batch(batch_indices_start=self._trav, batch_size=self._batch_size, epoch_end=self._epoch_end, array_length=self._max, current_epoch=self._epoch, total_epochs=self._num_epochs)
        self._trav = (integer_indexes[-1] + 1) % self._max
        result = self._dataframe.iloc[integer_indexes]
        cols = [result[col].values for col in result.columns]
        feed_dict = dict(zip(self._col_placeholders, cols))
        feed_dict[self._index_placeholder] = result.index.values
        return feed_dict