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
@pytest.mark.parametrize('degree', [3, 5])
def test_spline_transformer_periodic_splines_smoothness(degree):
    """Test that spline transformation is smooth at first / last knot."""
    X = np.linspace(-2, 10, 10000)[:, None]
    transformer = SplineTransformer(degree=degree, extrapolation='periodic', knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]])
    Xt = transformer.fit_transform(X)
    delta = (X.max() - X.min()) / len(X)
    tol = 10 * delta
    dXt = Xt
    for d in range(1, degree + 1):
        diff = np.diff(dXt, axis=0)
        assert np.abs(diff).max() < tol
        dXt = diff / delta
    diff = np.diff(dXt, axis=0)
    assert np.abs(diff).max() > 1