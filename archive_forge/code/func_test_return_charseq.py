import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_return_charseq(self):
    pyfunc = get_charseq
    cfunc = self.get_cfunc(pyfunc)
    for i in range(self.refsample1d.size):
        expected = pyfunc(self.refsample1d, i)
        got = cfunc(self.nbsample1d, i)
        self.assertEqual(expected, got)