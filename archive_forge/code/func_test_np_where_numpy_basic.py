from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_np_where_numpy_basic(self):
    pyfunc = np_where_3
    cfunc = jit(nopython=True)(pyfunc)
    dts = [bool, np.int16, np.int32, np.int64, np.double, np.complex128]
    for dt in dts:
        c = np.ones(53, dtype=bool)
        np.testing.assert_equal(cfunc(c, dt(0), dt(1)), dt(0))
        np.testing.assert_equal(cfunc(~c, dt(0), dt(1)), dt(1))
        np.testing.assert_equal(cfunc(True, dt(0), dt(1)), dt(0))
        np.testing.assert_equal(cfunc(False, dt(0), dt(1)), dt(1))
        d = np.ones_like(c).astype(dt)
        e = np.zeros_like(d)
        r = d.astype(dt)
        c[7] = False
        r[7] = e[7]
        np.testing.assert_equal(cfunc(c, e, e), e)
        np.testing.assert_equal(cfunc(c, d, e), r)
        np.testing.assert_equal(cfunc(c, d, e[0]), r)
        np.testing.assert_equal(cfunc(c, d[0], e), r)
        np.testing.assert_equal(cfunc(c[::2], d[::2], e[::2]), r[::2])
        np.testing.assert_equal(cfunc(c[1::2], d[1::2], e[1::2]), r[1::2])
        np.testing.assert_equal(cfunc(c[::3], d[::3], e[::3]), r[::3])
        np.testing.assert_equal(cfunc(c[1::3], d[1::3], e[1::3]), r[1::3])
        np.testing.assert_equal(cfunc(c[::-2], d[::-2], e[::-2]), r[::-2])
        np.testing.assert_equal(cfunc(c[::-3], d[::-3], e[::-3]), r[::-3])
        np.testing.assert_equal(cfunc(c[1::-3], d[1::-3], e[1::-3]), r[1::-3])