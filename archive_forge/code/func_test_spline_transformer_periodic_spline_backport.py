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
def test_spline_transformer_periodic_spline_backport():
    """Test that the backport of extrapolate="periodic" works correctly"""
    X = np.linspace(-2, 3.5, 10)[:, None]
    degree = 2
    transformer = SplineTransformer(degree=degree, extrapolation='periodic', knots=[[-1.0], [0.0], [1.0]])
    Xt = transformer.fit_transform(X)
    coef = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    spl = BSpline(np.arange(-3, 4), coef, degree, 'periodic')
    Xspl = spl(X[:, 0])
    assert_allclose(Xt, Xspl)