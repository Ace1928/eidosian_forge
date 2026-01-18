import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_unary_ufunc_where_same(self):
    ufunc = np.invert

    def check(a, out, mask):
        c1 = ufunc(a, out=out.copy(), where=mask.copy())
        c2 = ufunc(a, out=out, where=mask)
        assert_array_equal(c1, c2)
    x = np.arange(100).astype(np.bool_)
    check(x, x, x)
    check(x, x.copy(), x)
    check(x, x, x.copy())