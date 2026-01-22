from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
def _reset(self, capacity=20):
    """Resets internal variables."""
    if capacity <= 0:
        raise ValueError('IterationCountEstimator `capacity` must be positive. Actual:%d.' % capacity)
    self._buffer_wheel = collections.deque([])
    self._capacity = capacity
    self._min_iterations = 1
    self._last_iterations = self._min_iterations
    self._sample_count = 0