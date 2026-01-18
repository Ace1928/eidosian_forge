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
def test_pls_regression_fit_1d_y():
    """Check that when fitting with 1d `y`, prediction should also be 1d.

    Non-regression test for Issue #26549.
    """
    X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    y = np.array([2, 6, 12, 20, 30, 42])
    expected = y.copy()
    plsr = PLSRegression().fit(X, y)
    y_pred = plsr.predict(X)
    assert y_pred.shape == expected.shape
    lr = LinearRegression().fit(X, y)
    vr = VotingRegressor([('lr', lr), ('plsr', plsr)])
    y_pred = vr.fit(X, y).predict(X)
    assert y_pred.shape == expected.shape
    assert_allclose(y_pred, expected)