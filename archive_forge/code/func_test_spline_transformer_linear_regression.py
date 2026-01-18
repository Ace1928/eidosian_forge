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
def test_spline_transformer_linear_regression(bias, intercept):
    """Test that B-splines fit a sinusodial curve pretty well."""
    X = np.linspace(0, 10, 100)[:, None]
    y = np.sin(X[:, 0]) + 2
    pipe = Pipeline(steps=[('spline', SplineTransformer(n_knots=15, degree=3, include_bias=bias, extrapolation='constant')), ('ols', LinearRegression(fit_intercept=intercept))])
    pipe.fit(X, y)
    assert_allclose(pipe.predict(X), y, rtol=0.001)