import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_return_in_catch(self):

    @njit
    def udt(x):
        try:
            print('A')
            if x:
                raise ZeroDivisionError
            print('B')
            r = 123
        except Exception:
            print('C')
            r = 321
            return r
        print('D')
        return r
    with captured_stdout() as stdout:
        res = udt(True)
    self.assertEqual(stdout.getvalue().split(), ['A', 'C'])
    self.assertEqual(res, 321)
    with captured_stdout() as stdout:
        res = udt(False)
    self.assertEqual(stdout.getvalue().split(), ['A', 'B', 'D'])
    self.assertEqual(res, 123)