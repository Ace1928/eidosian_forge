import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
class TestIsValidDM:

    def test_is_valid_dm_improper_shape_1D_E(self):
        D = np.zeros((5,), dtype=np.float64)
        with pytest.raises(ValueError):
            is_valid_dm_throw(D)

    def test_is_valid_dm_improper_shape_1D_F(self):
        D = np.zeros((5,), dtype=np.float64)
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_improper_shape_3D_E(self):
        D = np.zeros((3, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            is_valid_dm_throw(D)

    def test_is_valid_dm_improper_shape_3D_F(self):
        D = np.zeros((3, 3, 3), dtype=np.float64)
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_nonzero_diagonal_E(self):
        y = np.random.rand(10)
        D = squareform(y)
        for i in range(0, 5):
            D[i, i] = 2.0
        with pytest.raises(ValueError):
            is_valid_dm_throw(D)

    def test_is_valid_dm_nonzero_diagonal_F(self):
        y = np.random.rand(10)
        D = squareform(y)
        for i in range(0, 5):
            D[i, i] = 2.0
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_asymmetric_E(self):
        y = np.random.rand(10)
        D = squareform(y)
        D[1, 3] = D[3, 1] + 1
        with pytest.raises(ValueError):
            is_valid_dm_throw(D)

    def test_is_valid_dm_asymmetric_F(self):
        y = np.random.rand(10)
        D = squareform(y)
        D[1, 3] = D[3, 1] + 1
        assert_equal(is_valid_dm(D), False)

    def test_is_valid_dm_correct_1_by_1(self):
        D = np.zeros((1, 1), dtype=np.float64)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_2_by_2(self):
        y = np.random.rand(1)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_3_by_3(self):
        y = np.random.rand(3)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_4_by_4(self):
        y = np.random.rand(6)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)

    def test_is_valid_dm_correct_5_by_5(self):
        y = np.random.rand(10)
        D = squareform(y)
        assert_equal(is_valid_dm(D), True)