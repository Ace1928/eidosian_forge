import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_pass(self):

    @njit
    def foo(x):
        try:
            pass
        except:
            pass
        return x
    res = foo(123)
    self.assertEqual(res, 123)