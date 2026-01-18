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
@pytest.mark.skipif(sp_version < parse_version('1.8.0'), reason='The option `sparse_output` is available as of scipy 1.8.0')
@pytest.mark.parametrize('degree', range(1, 3))
@pytest.mark.parametrize('knots', ['uniform', 'quantile'])
@pytest.mark.parametrize('extrapolation', ['error', 'constant', 'linear', 'continue', 'periodic'])
@pytest.mark.parametrize('include_bias', [False, True])
def test_spline_transformer_sparse_output(degree, knots, extrapolation, include_bias, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(200).reshape(40, 5)
    splt_dense = SplineTransformer(degree=degree, knots=knots, extrapolation=extrapolation, include_bias=include_bias, sparse_output=False)
    splt_sparse = SplineTransformer(degree=degree, knots=knots, extrapolation=extrapolation, include_bias=include_bias, sparse_output=True)
    splt_dense.fit(X)
    splt_sparse.fit(X)
    X_trans_sparse = splt_sparse.transform(X)
    X_trans_dense = splt_dense.transform(X)
    assert sparse.issparse(X_trans_sparse) and X_trans_sparse.format == 'csr'
    assert_allclose(X_trans_dense, X_trans_sparse.toarray())
    X_min = np.amin(X, axis=0)
    X_max = np.amax(X, axis=0)
    X_extra = np.r_[np.linspace(X_min - 5, X_min, 10), np.linspace(X_max, X_max + 5, 10)]
    if extrapolation == 'error':
        msg = 'X contains values beyond the limits of the knots'
        with pytest.raises(ValueError, match=msg):
            splt_dense.transform(X_extra)
        msg = 'Out of bounds'
        with pytest.raises(ValueError, match=msg):
            splt_sparse.transform(X_extra)
    else:
        assert_allclose(splt_dense.transform(X_extra), splt_sparse.transform(X_extra).toarray())