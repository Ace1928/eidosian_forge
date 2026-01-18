import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._csr_polynomial_expansion import (
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('degree', range(1, 5))
@pytest.mark.parametrize('n_knots', range(3, 5))
@pytest.mark.parametrize('knots', ['uniform', 'quantile'])
@pytest.mark.parametrize('extrapolation', ['constant', 'periodic'])
def test_spline_transformer_unity_decomposition(degree, n_knots, knots, extrapolation):
    """Test that B-splines are indeed a decomposition of unity.

    Splines basis functions must sum up to 1 per row, if we stay in between boundaries.
    """
    X = np.linspace(0, 1, 100)[:, None]
    X_train = np.r_[[[0]], X[::2, :], [[1]]]
    X_test = X[1::2, :]
    if extrapolation == 'periodic':
        n_knots = n_knots + degree
    splt = SplineTransformer(n_knots=n_knots, degree=degree, knots=knots, include_bias=True, extrapolation=extrapolation)
    splt.fit(X_train)
    for X in [X_train, X_test]:
        assert_allclose(np.sum(splt.transform(X), axis=1), 1)