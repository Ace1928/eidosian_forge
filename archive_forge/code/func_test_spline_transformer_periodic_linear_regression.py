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
@pytest.mark.parametrize(['bias', 'intercept'], [(True, False), (False, True)])
def test_spline_transformer_periodic_linear_regression(bias, intercept):
    """Test that B-splines fit a periodic curve pretty well."""

    def f(x):
        return np.sin(2 * np.pi * x) - np.sin(8 * np.pi * x) + 3
    X = np.linspace(0, 1, 101)[:, None]
    pipe = Pipeline(steps=[('spline', SplineTransformer(n_knots=20, degree=3, include_bias=bias, extrapolation='periodic')), ('ols', LinearRegression(fit_intercept=intercept))])
    pipe.fit(X, f(X[:, 0]))
    X_ = np.linspace(-1, 2, 301)[:, None]
    predictions = pipe.predict(X_)
    assert_allclose(predictions, f(X_[:, 0]), atol=0.01, rtol=0.01)
    assert_allclose(predictions[0:100], predictions[100:200], rtol=0.001)