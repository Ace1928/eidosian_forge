import numba
import numpy as np
import sys
import itertools
import gc
from numba import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np.random.generator_methods import _get_proper_func
from numba.np.random.generator_core import next_uint32, next_uint64, next_double
from numpy.random import MT19937, Generator
from numba.core.errors import TypingError
from numba.tests.support import run_in_new_process_caching, SerialMixin
def test_permutation_exception(self):
    self.disable_leak_check()

    def dist_func(x, arr, axis):
        return x.permutation(arr, axis=axis)
    self._check_invalid_types(dist_func, ['x', 'axis'], [np.array([3, 4, 5]), 0], ['x', 'x'])
    rng = np.random.default_rng(1)
    with self.assertRaises(IndexError) as raises:
        numba.njit(dist_func)(rng, np.array([3, 4, 5]), 2)
    self.assertIn('Axis is out of bounds for the given array', str(raises.exception))
    with self.assertRaises(IndexError) as raises:
        numba.njit(dist_func)(rng, np.array([3, 4, 5]), -2)
    self.assertIn('Axis is out of bounds for the given array', str(raises.exception))