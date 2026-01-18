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
def test_spline_transformer_periodic_splines_periodicity():
    """Test if shifted knots result in the same transformation up to permutation."""
    X = np.linspace(0, 10, 101)[:, None]
    transformer_1 = SplineTransformer(degree=3, extrapolation='periodic', knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]])
    transformer_2 = SplineTransformer(degree=3, extrapolation='periodic', knots=[[1.0], [3.0], [4.0], [5.0], [8.0], [9.0]])
    Xt_1 = transformer_1.fit_transform(X)
    Xt_2 = transformer_2.fit_transform(X)
    assert_allclose(Xt_1, Xt_2[:, [4, 0, 1, 2, 3]])