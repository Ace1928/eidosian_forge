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
@pytest.mark.parametrize(['deg', 'include_bias', 'interaction_only', 'dtype'], [(1, True, False, int), (2, True, False, int), (2, True, False, np.float32), (2, True, False, np.float64), (3, False, False, np.float64), (3, False, True, np.float64), (4, False, False, np.float64), (4, False, True, np.float64)])
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_polynomial_features_csc_X(deg, include_bias, interaction_only, dtype, csc_container):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, (100, 2))
    X_csc = csc_container(X)
    est = PolynomialFeatures(deg, include_bias=include_bias, interaction_only=interaction_only)
    Xt_csc = est.fit_transform(X_csc.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype))
    assert sparse.issparse(Xt_csc) and Xt_csc.format == 'csc'
    assert Xt_csc.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csc.toarray(), Xt_dense)