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
@pytest.mark.parametrize(['knots', 'n_knots', 'sample_weight', 'expected_knots'], [('uniform', 3, None, np.array([[0, 2], [3, 8], [6, 14]])), ('uniform', 3, np.array([0, 0, 1, 1, 0, 3, 1]), np.array([[2, 2], [4, 8], [6, 14]])), ('uniform', 4, None, np.array([[0, 2], [2, 6], [4, 10], [6, 14]])), ('quantile', 3, None, np.array([[0, 2], [3, 3], [6, 14]])), ('quantile', 3, np.array([0, 0, 1, 1, 0, 3, 1]), np.array([[2, 2], [5, 8], [6, 14]]))])
def test_spline_transformer_get_base_knot_positions(knots, n_knots, sample_weight, expected_knots):
    """Check the behaviour to find knot positions with and without sample_weight."""
    X = np.array([[0, 2], [0, 2], [2, 2], [3, 3], [4, 6], [5, 8], [6, 14]])
    base_knots = SplineTransformer._get_base_knot_positions(X=X, knots=knots, n_knots=n_knots, sample_weight=sample_weight)
    assert_allclose(base_knots, expected_knots)