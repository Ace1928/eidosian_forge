import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_print_multiple_values(self):
    pyfunc = print_values
    cfunc = njit((types.intp,) * 3)(pyfunc)
    with captured_stdout():
        cfunc(1, 2, 3)
        self.assertEqual(sys.stdout.getvalue(), '1 2 3\n')