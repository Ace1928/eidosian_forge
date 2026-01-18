import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_in_prange_map(self):

    def udt(arr, x):
        out = arr.copy()
        for i in prange(arr.size):
            try:
                if i == x:
                    raise ValueError
                out[i] = arr[i] + i
            except Exception:
                out[i] = -1
        return out
    args = [np.arange(10), 6]
    expect = udt(*args)
    self.assertPreciseEqual(njit(parallel=False)(udt)(*args), expect)
    self.assertPreciseEqual(njit(parallel=True)(udt)(*args), expect)