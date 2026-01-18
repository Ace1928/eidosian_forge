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
@pytest.mark.parametrize('n_samples, n_features', [(100, 10), (100, 200)])
def test_singular_value_helpers(n_samples, n_features, global_random_seed):
    X, Y = make_regression(n_samples, n_features, n_targets=5, random_state=global_random_seed)
    u1, v1, _ = _get_first_singular_vectors_power_method(X, Y, norm_y_weights=True)
    u2, v2 = _get_first_singular_vectors_svd(X, Y)
    _svd_flip_1d(u1, v1)
    _svd_flip_1d(u2, v2)
    rtol = 0.001
    assert_allclose(u1, u2, atol=u2.max() * rtol)
    assert_allclose(v1, v2, atol=v2.max() * rtol)