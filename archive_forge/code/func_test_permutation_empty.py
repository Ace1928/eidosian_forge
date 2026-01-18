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
def test_permutation_empty(self):
    a = np.array([])
    b = np.array([])

    def dist_func(x, arr):
        return x.permutation(arr)
    nb_func = numba.njit(dist_func)
    rng = lambda: np.random.default_rng(1)
    self.assertPreciseEqual(dist_func(rng(), a), nb_func(rng(), b))