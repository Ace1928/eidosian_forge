import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def test_write_input_arg(self):

    @guvectorize(['void(float64[:], uint8[:])'], '(n)->(n)')
    def func(x, out):
        for i in range(x.size):
            if i % 4 == 0:
                out[i] = 1
    x = np.random.rand(10, 5)
    out = np.zeros_like(x, dtype=np.int8)
    func(x, out)
    np.testing.assert_array_equal(np.array([True, False, False, False, True], dtype=np.bool_), out.any(axis=0))