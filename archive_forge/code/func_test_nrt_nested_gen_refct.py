import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
@unittest.expectedFailure
def test_nrt_nested_gen_refct(self):

    def gen0(arr):
        yield arr

    def factory(gen0):

        def gen1(arr):
            for out in gen0(arr):
                return out
        return gen1
    py_arr = np.arange(10)
    c_arr = py_arr.copy()
    py_old = factory(gen0)(py_arr)
    c_gen = jit(nopython=True)(factory(jit(nopython=True)(gen0)))
    c_old = c_gen(c_arr)
    self.assertIsNot(py_arr, c_arr)
    self.assertIs(py_old, py_arr)
    self.assertIs(c_old, c_arr)
    self.assertRefCountEqual(py_old, c_old)