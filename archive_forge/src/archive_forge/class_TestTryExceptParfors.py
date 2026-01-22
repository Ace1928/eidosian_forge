import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
@skip_parfors_unsupported
class TestTryExceptParfors(TestCase):

    def test_try_in_prange_reduction(self):

        def udt(n):
            c = 0
            for i in prange(n):
                try:
                    c += 1
                except Exception:
                    c += 1
            return c
        args = [10]
        expect = udt(*args)
        self.assertEqual(njit(parallel=False)(udt)(*args), expect)
        self.assertEqual(njit(parallel=True)(udt)(*args), expect)

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

    def test_try_outside_prange_map(self):

        def udt(arr, x):
            out = arr.copy()
            try:
                for i in prange(arr.size):
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