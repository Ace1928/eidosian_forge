import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
@pytest.mark.parametrize('bsr_container', BSR_CONTAINERS)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_pairwise_distances_for_sparse_data(coo_container, csc_container, bsr_container, csr_container, global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    Y = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    X_sparse = csr_container(X)
    Y_sparse = csr_container(Y)
    S = pairwise_distances(X_sparse, Y_sparse, metric='euclidean')
    S2 = euclidean_distances(X_sparse, Y_sparse)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype
    S = pairwise_distances(X_sparse, Y_sparse, metric='cosine')
    S2 = cosine_distances(X_sparse, Y_sparse)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype
    S = pairwise_distances(X_sparse, csc_container(Y), metric='manhattan')
    S2 = manhattan_distances(bsr_container(X), coo_container(Y))
    assert_allclose(S, S2)
    if global_dtype == np.float64:
        assert S.dtype == S2.dtype == global_dtype
    else:
        with pytest.raises(AssertionError):
            assert S.dtype == S2.dtype == global_dtype
    S2 = manhattan_distances(X, Y)
    assert_allclose(S, S2)
    if global_dtype == np.float64:
        assert S.dtype == S2.dtype == global_dtype
    else:
        with pytest.raises(AssertionError):
            assert S.dtype == S2.dtype == global_dtype
    kwds = {'p': 2.0}
    S = pairwise_distances(X, Y, metric='minkowski', **kwds)
    S2 = pairwise_distances(X, Y, metric=minkowski, **kwds)
    assert_allclose(S, S2)
    kwds = {'p': 2.0}
    S = pairwise_distances(X, metric='minkowski', **kwds)
    S2 = pairwise_distances(X, metric=minkowski, **kwds)
    assert_allclose(S, S2)
    with pytest.raises(TypeError):
        pairwise_distances(X_sparse, metric='minkowski')
    with pytest.raises(TypeError):
        pairwise_distances(X, Y_sparse, metric='minkowski')