import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_state_reset(self):

    @njit
    def inner(x):
        if x == 1:
            raise MyError('one')
        elif x == 2:
            raise MyError('two')

    @njit
    def udt(x):
        try:
            inner(x)
            res = 'not raised'
        except:
            res = 'caught'
        if x == 0:
            inner(2)
        return res
    with self.assertRaises(MyError) as raises:
        udt(0)
    self.assertEqual(str(raises.exception), 'two')
    self.assertEqual(udt(1), 'caught')
    self.assertEqual(udt(-1), 'not raised')