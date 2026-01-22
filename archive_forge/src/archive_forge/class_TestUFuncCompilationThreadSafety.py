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
class TestUFuncCompilationThreadSafety(TestCase):

    def test_lock(self):
        """
        Test that (lazy) compiling from several threads at once doesn't
        produce errors (see issue #2403).
        """
        errors = []

        @vectorize
        def foo(x):
            return x + 1

        def wrapper():
            try:
                a = np.ones((10,), dtype=np.float64)
                expected = np.ones((10,), dtype=np.float64) + 1.0
                np.testing.assert_array_equal(foo(a), expected)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=wrapper) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)