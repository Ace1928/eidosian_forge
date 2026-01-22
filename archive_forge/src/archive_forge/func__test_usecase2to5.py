import sys
import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.tests.support import captured_stdout, TestCase
from numba.np import numpy_support
def _test_usecase2to5(self, pyfunc, dtype):
    array = self._setup_usecase2to5(dtype)
    record_type = numpy_support.from_dtype(dtype)
    cfunc = njit((record_type[:], types.intp))(pyfunc)
    with captured_stdout():
        pyfunc(array, len(array))
        expect = sys.stdout.getvalue()
    with captured_stdout():
        cfunc(array, len(array))
        got = sys.stdout.getvalue()
    self.assertEqual(expect, got)