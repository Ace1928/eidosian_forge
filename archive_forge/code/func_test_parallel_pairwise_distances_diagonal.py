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
@pytest.mark.parametrize('metric', ('euclidean', 'l2', 'sqeuclidean'))
def test_parallel_pairwise_distances_diagonal(metric, global_dtype):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 10), scale=10000000000.0).astype(global_dtype, copy=False)
    distances = pairwise_distances(X, metric=metric, n_jobs=2)
    assert_allclose(np.diag(distances), 0, atol=1e-10)