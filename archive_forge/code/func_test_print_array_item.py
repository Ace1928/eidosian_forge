import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_print_array_item(self):
    """
        Test printing a Numpy character sequence
        """
    dtype = np.dtype([('x', 'S4')])
    arr = np.frombuffer(bytearray(range(1, 9)), dtype=dtype)
    pyfunc = print_array_item
    cfunc = jit(nopython=True)(pyfunc)
    for i in range(len(arr)):
        with captured_stdout():
            cfunc(arr, i)
            self.assertEqual(sys.stdout.getvalue(), str(arr[i]['x']) + '\n')