import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_print_empty(self):
    pyfunc = print_empty
    cfunc = njit(())(pyfunc)
    with captured_stdout():
        cfunc()
        self.assertEqual(sys.stdout.getvalue(), '\n')