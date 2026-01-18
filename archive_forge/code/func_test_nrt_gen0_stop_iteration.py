import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_nrt_gen0_stop_iteration(self):
    """
        Test cleanup on StopIteration
        """
    pygen = nrt_gen0
    cgen = jit(nopython=True)(pygen)
    py_ary = np.arange(1)
    c_ary = py_ary.copy()
    py_iter = pygen(py_ary)
    c_iter = cgen(c_ary)
    py_res = next(py_iter)
    c_res = next(c_iter)
    with self.assertRaises(StopIteration):
        py_res = next(py_iter)
    with self.assertRaises(StopIteration):
        c_res = next(c_iter)
    del py_iter
    del c_iter
    np.testing.assert_equal(py_ary, c_ary)
    self.assertEqual(py_res, c_res)
    self.assertRefCountEqual(py_ary, c_ary)