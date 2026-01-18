import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_ufunc_at_manual(self):

    def check(ufunc, a, ind, b=None):
        a0 = a.copy()
        if b is None:
            ufunc.at(a0, ind.copy())
            c1 = a0.copy()
            ufunc.at(a, ind)
            c2 = a.copy()
        else:
            ufunc.at(a0, ind.copy(), b.copy())
            c1 = a0.copy()
            ufunc.at(a, ind, b)
            c2 = a.copy()
        assert_array_equal(c1, c2)
    a = np.arange(10000, dtype=np.int16)
    check(np.invert, a[::-1], a)
    a = np.arange(100, dtype=np.int16)
    ind = np.arange(0, 100, 2, dtype=np.int16)
    check(np.add, a, ind, a[25:75])