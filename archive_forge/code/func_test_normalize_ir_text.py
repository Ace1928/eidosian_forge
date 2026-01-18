from numba import njit, cfunc
from numba.tests.support import TestCase, unittest
from numba.core import cgutils
def test_normalize_ir_text(self):
    out = cgutils.normalize_ir_text('abc')
    self.assertIsInstance(out, str)
    out.encode('latin1')