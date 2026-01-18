import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_outside_prange_reduction(self):

    def udt(n):
        c = 0
        try:
            for i in prange(n):
                c += 1
        except Exception:
            return 57005
        else:
            return c
    args = [10]
    expect = udt(*args)
    self.assertEqual(njit(parallel=False)(udt)(*args), expect)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', NumbaPerformanceWarning)
        self.assertEqual(njit(parallel=True)(udt)(*args), expect)
    self.assertEqual(len(w), 1)
    self.assertIn('no transformation for parallel execution was possible', str(w[0]))