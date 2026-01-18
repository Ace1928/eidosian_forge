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
@pytest.mark.parametrize('n_jobs', [1, 2])
@pytest.mark.parametrize('metric', ['seuclidean', 'mahalanobis'])
@pytest.mark.parametrize('dist_function', [pairwise_distances, pairwise_distances_chunked])
def test_pairwise_distances_data_derived_params(n_jobs, metric, dist_function):
    with config_context(working_memory=0.1):
        rng = np.random.RandomState(0)
        X = rng.random_sample((100, 10))
        expected_dist = squareform(pdist(X, metric=metric))
        dist = np.vstack(tuple(dist_function(X, metric=metric, n_jobs=n_jobs)))
        assert_allclose(dist, expected_dist)