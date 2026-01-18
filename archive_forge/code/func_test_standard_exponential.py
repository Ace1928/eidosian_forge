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
def test_standard_exponential(self):
    test_sizes = [None, (), (100,), (10, 20, 30)]
    test_dtypes = [np.float32, np.float64]
    bitgen_types = [None, MT19937]
    dist_func = lambda x, size, dtype: x.standard_exponential()
    with self.subTest():
        self.check_numpy_parity(dist_func, test_size=None, test_dtype=None)
    dist_func = lambda x, size, dtype: x.standard_exponential(size=size, dtype=dtype)
    for _size in test_sizes:
        for _dtype in test_dtypes:
            for _bitgen in bitgen_types:
                with self.subTest(_size=_size, _dtype=_dtype, _bitgen=_bitgen):
                    self.check_numpy_parity(dist_func, _bitgen, None, _size, _dtype)
    dist_func = lambda x, method, size, dtype: x.standard_exponential(method=method, size=size, dtype=dtype)
    self._check_invalid_types(dist_func, ['method', 'size', 'dtype'], ['zig', (1,), np.float32], [0, ('x',), 0])