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
def test_proper_func_provider(self):

    def test_32bit_func():
        return 32

    def test_64bit_func():
        return 64
    self.assertEqual(_get_proper_func(test_32bit_func, test_64bit_func, np.float64)[0](), 64)
    self.assertEqual(_get_proper_func(test_32bit_func, test_64bit_func, np.float32)[0](), 32)
    with self.assertRaises(TypingError) as raises:
        _get_proper_func(test_32bit_func, test_64bit_func, np.int32)
    self.assertIn('Argument dtype is not one of the expected type(s)', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        _get_proper_func(test_32bit_func, test_64bit_func, types.float64)
    self.assertIn('Argument dtype is not one of the expected type(s)', str(raises.exception))