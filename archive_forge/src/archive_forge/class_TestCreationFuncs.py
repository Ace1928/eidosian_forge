import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestCreationFuncs:

    def setup_method(self):
        dtypes = {np.dtype(tp) for tp in itertools.chain(*np.sctypes.values())}
        variable_sized = {tp for tp in dtypes if tp.str.endswith('0')}
        self.dtypes = sorted(dtypes - variable_sized | {np.dtype(tp.str.replace('0', str(i))) for tp in variable_sized for i in range(1, 10)}, key=lambda dtype: dtype.str)
        self.orders = {'C': 'c_contiguous', 'F': 'f_contiguous'}
        self.ndims = 10

    def check_function(self, func, fill_value=None):
        par = ((0, 1, 2), range(self.ndims), self.orders, self.dtypes)
        fill_kwarg = {}
        if fill_value is not None:
            fill_kwarg = {'fill_value': fill_value}
        for size, ndims, order, dtype in itertools.product(*par):
            shape = ndims * [size]
            if fill_kwarg and dtype.str.startswith('|V'):
                continue
            arr = func(shape, order=order, dtype=dtype, **fill_kwarg)
            assert_equal(arr.dtype, dtype)
            assert_(getattr(arr.flags, self.orders[order]))
            if fill_value is not None:
                if dtype.str.startswith('|S'):
                    val = str(fill_value)
                else:
                    val = fill_value
                assert_equal(arr, dtype.type(val))

    def test_zeros(self):
        self.check_function(np.zeros)

    def test_ones(self):
        self.check_function(np.ones)

    def test_empty(self):
        self.check_function(np.empty)

    def test_full(self):
        self.check_function(np.full, 0)
        self.check_function(np.full, 1)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
    def test_for_reference_leak(self):
        dim = 1
        beg = sys.getrefcount(dim)
        np.zeros([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.ones([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.empty([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.full([dim] * 10, 0)
        assert_(sys.getrefcount(dim) == beg)