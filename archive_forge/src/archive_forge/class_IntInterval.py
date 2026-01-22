import abc
import hashlib
import json
import random
import time
import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
class IntInterval(Domain):
    """A domain that takes on all integer values in a closed interval."""

    def __init__(self, min_value=None, max_value=None):
        """Create an `IntInterval`.

        Args:
          min_value: The lower bound (inclusive) of the interval.
          max_value: The upper bound (inclusive) of the interval.

        Raises:
          TypeError: If `min_value` or `max_value` is not an `int`.
          ValueError: If `min_value > max_value`.
        """
        if not isinstance(min_value, int):
            raise TypeError('min_value must be an int: %r' % (min_value,))
        if not isinstance(max_value, int):
            raise TypeError('max_value must be an int: %r' % (max_value,))
        if min_value > max_value:
            raise ValueError('%r > %r' % (min_value, max_value))
        self._min_value = min_value
        self._max_value = max_value

    def __str__(self):
        return '[%s, %s]' % (self._min_value, self._max_value)

    def __repr__(self):
        return 'IntInterval(%r, %r)' % (self._min_value, self._max_value)

    @property
    def dtype(self):
        return int

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    def sample_uniform(self, rng=random):
        return rng.randint(self._min_value, self._max_value)

    def update_hparam_info(self, hparam_info):
        hparam_info.type = api_pb2.DATA_TYPE_FLOAT64
        hparam_info.domain_interval.min_value = self._min_value
        hparam_info.domain_interval.max_value = self._max_value