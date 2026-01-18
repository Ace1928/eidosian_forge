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
def test_npgen_boxing_unboxing(self):
    rng_instance = np.random.default_rng()
    numba_func = numba.njit(lambda x: x)
    self.assertEqual(rng_instance, numba_func(rng_instance))
    self.assertEqual(id(rng_instance), id(numba_func(rng_instance)))