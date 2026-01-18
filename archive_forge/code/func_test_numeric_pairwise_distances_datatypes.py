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
@pytest.mark.parametrize('metric', ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'hamming', 'mahalanobis', 'minkowski', 'seuclidean', 'sqeuclidean', 'cityblock', 'cosine', 'euclidean'])
@pytest.mark.parametrize('y_is_x', [True, False], ids=['Y is X', 'Y is not X'])
def test_numeric_pairwise_distances_datatypes(metric, global_dtype, y_is_x):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    params = {}
    if y_is_x:
        Y = X
        expected_dist = squareform(pdist(X, metric=metric))
    else:
        Y = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
        expected_dist = cdist(X, Y, metric=metric)
        if metric == 'seuclidean':
            params = {'V': np.var(np.vstack([X, Y]), axis=0, ddof=1, dtype=np.float64)}
        elif metric == 'mahalanobis':
            params = {'VI': np.linalg.inv(np.cov(np.vstack([X, Y]).T)).T}
    dist = pairwise_distances(X, Y, metric=metric, **params)
    assert_allclose(dist, expected_dist)