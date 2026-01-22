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
class TestIndices:

    def test_simple(self):
        [x, y] = np.indices((4, 3))
        assert_array_equal(x, np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]))
        assert_array_equal(y, np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]))

    def test_single_input(self):
        [x] = np.indices((4,))
        assert_array_equal(x, np.array([0, 1, 2, 3]))
        [x] = np.indices((4,), sparse=True)
        assert_array_equal(x, np.array([0, 1, 2, 3]))

    def test_scalar_input(self):
        assert_array_equal([], np.indices(()))
        assert_array_equal([], np.indices((), sparse=True))
        assert_array_equal([[]], np.indices((0,)))
        assert_array_equal([[]], np.indices((0,), sparse=True))

    def test_sparse(self):
        [x, y] = np.indices((4, 3), sparse=True)
        assert_array_equal(x, np.array([[0], [1], [2], [3]]))
        assert_array_equal(y, np.array([[0, 1, 2]]))

    @pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.parametrize('dims', [(), (0,), (4, 3)])
    def test_return_type(self, dtype, dims):
        inds = np.indices(dims, dtype=dtype)
        assert_(inds.dtype == dtype)
        for arr in np.indices(dims, dtype=dtype, sparse=True):
            assert_(arr.dtype == dtype)