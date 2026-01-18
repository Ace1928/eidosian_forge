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
def test_sanity_check_pls_regression():
    d = load_linnerud()
    X = d.data
    Y = d.target
    pls = PLSRegression(n_components=X.shape[1])
    X_trans, _ = pls.fit_transform(X, Y)
    assert_allclose(X_trans, pls.x_scores_)
    expected_x_weights = np.array([[-0.61330704, -0.00443647, 0.78983213], [-0.74697144, -0.32172099, -0.58183269], [-0.25668686, 0.94682413, -0.19399983]])
    expected_x_loadings = np.array([[-0.61470416, -0.24574278, 0.78983213], [-0.65625755, -0.14396183, -0.58183269], [-0.51733059, 1.00609417, -0.19399983]])
    expected_y_weights = np.array([[+0.32456184, 0.29892183, 0.20316322], [+0.42439636, 0.61970543, 0.19320542], [-0.13143144, -0.26348971, -0.17092916]])
    expected_y_loadings = np.array([[+0.32456184, 0.29892183, 0.20316322], [+0.42439636, 0.61970543, 0.19320542], [-0.13143144, -0.26348971, -0.17092916]])
    assert_array_almost_equal(np.abs(pls.x_loadings_), np.abs(expected_x_loadings))
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))
    x_loadings_sign_flip = np.sign(pls.x_loadings_ / expected_x_loadings)
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    y_loadings_sign_flip = np.sign(pls.y_loadings_ / expected_y_loadings)
    assert_array_almost_equal(x_loadings_sign_flip, x_weights_sign_flip)
    assert_array_almost_equal(y_loadings_sign_flip, y_weights_sign_flip)