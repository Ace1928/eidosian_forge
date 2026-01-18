import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_nrt_nested_nopython_gen(self):
    """
        Test nesting three generators
        """

    def factory(decor=lambda x: x):

        @decor
        def foo(a, n):
            for i in range(n):
                yield a[i]
                a[i] += i

        @decor
        def bar(n):
            a = np.arange(n)
            for i in foo(a, n):
                yield (i * 2)
            for i in range(a.size):
                yield a[i]

        @decor
        def cat(n):
            for i in bar(n):
                yield (i + i)
        return cat
    py_gen = factory()
    c_gen = factory(jit(nopython=True))
    py_res = list(py_gen(10))
    c_res = list(c_gen(10))
    self.assertEqual(py_res, c_res)