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
def test_univariate(self):
    np.random.seed(12345)
    for x in np.linspace(-5, 5, num=11):
        A = np.array([[x]])
        assert_allclose(expm_cond(A), abs(x))
    for x in np.logspace(-2, 2, num=11):
        A = np.array([[x]])
        assert_allclose(expm_cond(A), abs(x))
    for i in range(10):
        A = np.random.randn(1, 1)
        assert_allclose(expm_cond(A), np.absolute(A)[0, 0])