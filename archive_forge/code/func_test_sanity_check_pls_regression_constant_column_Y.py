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
def test_sanity_check_pls_regression_constant_column_Y():
    d = load_linnerud()
    X = d.data
    Y = d.target
    Y[:, 0] = 1
    pls = PLSRegression(n_components=X.shape[1])
    pls.fit(X, Y)
    expected_x_weights = np.array([[-0.6273573, 0.007081799, 0.7786994], [-0.7493417, -0.277612681, -0.6011807], [-0.2119194, 0.960666981, -0.179469]])
    expected_x_loadings = np.array([[-0.6273512, -0.22464538, 0.7786994], [-0.6643156, -0.09871193, -0.6011807], [-0.5125877, 1.0140738, -0.179469]])
    expected_y_loadings = np.array([[0.0, 0.0, 0.0], [0.43573, 0.5828479, 0.2174802], [-0.1353739, -0.2486423, -0.1810386]])
    assert_array_almost_equal(np.abs(expected_x_weights), np.abs(pls.x_weights_))
    assert_array_almost_equal(np.abs(expected_x_loadings), np.abs(pls.x_loadings_))
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_loadings))
    x_loadings_sign_flip = np.sign(expected_x_loadings / pls.x_loadings_)
    x_weights_sign_flip = np.sign(expected_x_weights / pls.x_weights_)
    y_loadings_sign_flip = np.sign(expected_y_loadings[1:] / pls.y_loadings_[1:])
    assert_array_equal(x_loadings_sign_flip, x_weights_sign_flip)
    assert_array_equal(x_loadings_sign_flip[1:], y_loadings_sign_flip)