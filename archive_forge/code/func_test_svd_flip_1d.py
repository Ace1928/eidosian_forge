import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.cross_decomposition._pls import (
from sklearn.datasets import load_linnerud, make_regression
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
def test_svd_flip_1d():
    u = np.array([1, -4, 2])
    v = np.array([1, 2, 3])
    u_expected, v_expected = svd_flip(u.reshape(-1, 1), v.reshape(1, -1))
    _svd_flip_1d(u, v)
    assert_allclose(u, u_expected.ravel())
    assert_allclose(u, [-1, 4, -2])
    assert_allclose(v, v_expected.ravel())
    assert_allclose(v, [-1, -2, -3])