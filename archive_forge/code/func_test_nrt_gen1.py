import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_nrt_gen1(self):
    pygen = nrt_gen1
    cgen = jit(nopython=True)(pygen)
    py_ary1 = np.arange(10)
    py_ary2 = py_ary1 + 100
    c_ary1 = py_ary1.copy()
    c_ary2 = py_ary2.copy()
    py_res = list(pygen(py_ary1, py_ary2))
    c_res = list(cgen(c_ary1, c_ary2))
    np.testing.assert_equal(py_ary1, c_ary1)
    np.testing.assert_equal(py_ary2, c_ary2)
    self.assertEqual(py_res, c_res)
    self.assertRefCountEqual(py_ary1, c_ary1)
    self.assertRefCountEqual(py_ary2, c_ary2)