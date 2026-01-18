import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_nrt_gen0_no_iter(self):
    """
        Test cleanup for a initialized but never iterated (never call next())
        generator.
        """
    pygen = nrt_gen0
    cgen = jit(nopython=True)(pygen)
    py_ary = np.arange(1)
    c_ary = py_ary.copy()
    py_iter = pygen(py_ary)
    c_iter = cgen(c_ary)
    del py_iter
    del c_iter
    np.testing.assert_equal(py_ary, c_ary)
    self.assertRefCountEqual(py_ary, c_ary)