import logging
import threading
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import TensorStructType
from ray.rllib.utils.serialization import _serialize_ndarray, _deserialize_ndarray
from ray.rllib.utils.deprecation import deprecation_warning
@DeveloperAPI
class RunningStat:

    def __init__(self, shape=()):
        self.num_pushes = 0
        self.mean_array = np.zeros(shape)
        self.std_array = np.zeros(shape)

    def copy(self):
        other = RunningStat()
        other.num_pushes = self.num_pushes if hasattr(self, 'num_pushes') else self._n
        other.mean_array = np.copy(self.mean_array) if hasattr(self, 'mean_array') else np.copy(self._M)
        other.std_array = np.copy(self.std_array) if hasattr(self, 'std_array') else np.copy(self._S)
        return other

    def push(self, x):
        x = np.asarray(x)
        if x.shape != self.mean_array.shape:
            raise ValueError('Unexpected input shape {}, expected {}, value = {}'.format(x.shape, self.mean_array.shape, x))
        self.num_pushes += 1
        if self.num_pushes == 1:
            self.mean_array[...] = x
        else:
            delta = x - self.mean_array
            self.mean_array[...] += delta / self.num_pushes
            self.std_array[...] += delta * delta * (self.num_pushes - 1) / self.num_pushes

    def update(self, other):
        n1 = self.num_pushes
        n2 = other.num_pushes
        n = n1 + n2
        if n == 0:
            return
        delta = self.mean_array - other.mean_array
        delta2 = delta * delta
        m = (n1 * self.mean_array + n2 * other.mean_array) / n
        s = self.std_array + other.std_array + delta2 * n1 * n2 / n
        self.num_pushes = n
        self.mean_array = m
        self.std_array = s

    def __repr__(self):
        return '(n={}, mean_mean={}, mean_std={})'.format(self.n, np.mean(self.mean), np.mean(self.std))

    @property
    def n(self):
        return self.num_pushes

    @property
    def mean(self):
        return self.mean_array

    @property
    def var(self):
        return self.std_array / (self.num_pushes - 1) if self.num_pushes > 1 else np.square(self.mean_array)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self.mean_array.shape

    def to_state(self):
        return {'num_pushes': self.num_pushes, 'mean_array': _serialize_ndarray(self.mean_array), 'std_array': _serialize_ndarray(self.std_array)}

    @staticmethod
    def from_state(state):
        running_stats = RunningStat()
        running_stats.num_pushes = state['num_pushes']
        running_stats.mean_array = _deserialize_ndarray(state['mean_array'])
        running_stats.std_array = _deserialize_ndarray(state['std_array'])
        return running_stats