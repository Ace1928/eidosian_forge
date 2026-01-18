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
def test_pairwise_distances_chunked_diagonal(metric, global_dtype):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 10), scale=10000000000.0).astype(global_dtype, copy=False)
    chunks = list(pairwise_distances_chunked(X, working_memory=1, metric=metric))
    assert len(chunks) > 1
    assert_allclose(np.diag(np.vstack(chunks)), 0, rtol=1e-10)