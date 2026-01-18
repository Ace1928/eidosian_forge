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
def test_spline_transformer_kbindiscretizer():
    """Test that a B-spline of degree=0 is equivalent to KBinsDiscretizer."""
    rng = np.random.RandomState(97531)
    X = rng.randn(200).reshape(200, 1)
    n_bins = 5
    n_knots = n_bins + 1
    splt = SplineTransformer(n_knots=n_knots, degree=0, knots='quantile', include_bias=True)
    splines = splt.fit_transform(X)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile')
    kbins = kbd.fit_transform(X)
    assert_allclose(splines, kbins, rtol=1e-13)