import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_raise_in_try(self):

    @njit
    def udt(x):
        try:
            print('A')
            if x:
                raise MyError('my_error')
            print('B')
        except:
            print('C')
            return 321
        return 123
    with captured_stdout() as stdout:
        res = udt(True)
    self.assertEqual(stdout.getvalue().split(), ['A', 'C'])
    self.assertEqual(res, 321)
    with captured_stdout() as stdout:
        res = udt(False)
    self.assertEqual(stdout.getvalue().split(), ['A', 'B'])
    self.assertEqual(res, 123)