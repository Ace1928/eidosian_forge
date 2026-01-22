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
class TestHelperFuncs(TestCase):

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

    def test_check_types(self):
        rng = np.random.default_rng(1)
        py_func = lambda x: x.normal(loc=(0,))
        numba_func = numba.njit(cache=True)(py_func)
        with self.assertRaises(TypingError) as raises:
            numba_func(rng)
        self.assertIn('Argument loc is not one of the expected type(s): ' + "[<class 'numba.core.types.scalars.Float'>, " + "<class 'numba.core.types.scalars.Integer'>, " + "<class 'int'>, <class 'float'>]", str(raises.exception))

    def test_integers_arg_check(self):
        rng = np.random.default_rng(1)
        py_func = lambda x, low, high, dtype: x.integers(low=low, high=high, dtype=dtype, endpoint=True)
        numba_func = numba.njit()(py_func)
        numba_func_low = numba.njit()(py_func)
        py_func = lambda x, low, high, dtype: x.integers(low=low, high=high, dtype=dtype, endpoint=False)
        numba_func_endpoint_false = numba.njit()(py_func)
        cases = [(np.iinfo(np.uint8).min, np.iinfo(np.uint8).max, np.uint8), (np.iinfo(np.int8).min, np.iinfo(np.int8).max, np.int8), (np.iinfo(np.uint16).min, np.iinfo(np.uint16).max, np.uint16), (np.iinfo(np.int16).min, np.iinfo(np.int16).max, np.int16), (np.iinfo(np.uint32).min, np.iinfo(np.uint32).max, np.uint32), (np.iinfo(np.int32).min, np.iinfo(np.int32).max, np.int32)]
        for low, high, dtype in cases:
            with self.subTest(low=low, high=high, dtype=dtype):
                with self.assertRaises(ValueError) as raises:
                    numba_func_low(rng, low - 1, high, dtype)
                self.assertIn('low is out of bounds', str(raises.exception))
                with self.assertRaises(ValueError) as raises:
                    numba_func(rng, low, high + 1, dtype)
                self.assertIn('high is out of bounds', str(raises.exception))
                with self.assertRaises(ValueError) as raises:
                    numba_func_endpoint_false(rng, low, high + 2, dtype)
                self.assertIn('high is out of bounds', str(raises.exception))
        low, high, dtype = (np.iinfo(np.uint64).min, np.iinfo(np.uint64).max, np.uint64)
        with self.assertRaises(ValueError) as raises:
            numba_func_low(rng, low - 1, high, dtype)
        self.assertIn('low is out of bounds', str(raises.exception))
        low, high, dtype = (np.iinfo(np.int64).min, np.iinfo(np.int64).max, np.int64)
        with self.assertRaises(ValueError) as raises:
            numba_func(rng, low, high + 1, dtype)
        self.assertIn('high is out of bounds', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            numba_func_endpoint_false(rng, low, high + 2, dtype)
        self.assertIn('high is out of bounds', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            numba_func(rng, 105, 100, np.uint32)
        self.assertIn('low is greater than high in given interval', str(raises.exception))