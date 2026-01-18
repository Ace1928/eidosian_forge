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
def test_opposite_sign_complex_eigenvalues(self):
    M = [[2j, 4], [0, -2j]]
    R = [[1 + 1j, 2], [0, 1 - 1j]]
    assert_allclose(np.dot(R, R), M, atol=1e-14)
    assert_allclose(fractional_matrix_power(M, 0.5), R, atol=1e-14)