import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
class TestKhatriRao:

    def test_basic(self):
        a = khatri_rao(array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]]))
        assert_array_equal(a, array([[5, 12], [7, 16], [15, 24], [21, 32]]))
        b = khatri_rao(np.empty([2, 2]), np.empty([2, 2]))
        assert_array_equal(b.shape, (4, 2))

    def test_number_of_columns_equality(self):
        with pytest.raises(ValueError):
            a = array([[1, 2, 3], [4, 5, 6]])
            b = array([[1, 2], [3, 4]])
            khatri_rao(a, b)

    def test_to_assure_2d_array(self):
        with pytest.raises(ValueError):
            a = array([1, 2, 3])
            b = array([4, 5, 6])
            khatri_rao(a, b)
        with pytest.raises(ValueError):
            a = array([1, 2, 3])
            b = array([[1, 2, 3], [4, 5, 6]])
            khatri_rao(a, b)
        with pytest.raises(ValueError):
            a = array([[1, 2, 3], [7, 8, 9]])
            b = array([4, 5, 6])
            khatri_rao(a, b)

    def test_equality_of_two_equations(self):
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])
        res1 = khatri_rao(a, b)
        res2 = np.vstack([np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T
        assert_array_equal(res1, res2)