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
@pytest.mark.parametrize('PLSEstimator', [PLSRegression, PLSCanonical, CCA])
def test_pls_coef_shape(PLSEstimator):
    """Check the shape of `coef_` attribute.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12410
    """
    d = load_linnerud()
    X = d.data
    Y = d.target
    pls = PLSEstimator(copy=True).fit(X, Y)
    n_targets, n_features = (Y.shape[1], X.shape[1])
    assert pls.coef_.shape == (n_targets, n_features)