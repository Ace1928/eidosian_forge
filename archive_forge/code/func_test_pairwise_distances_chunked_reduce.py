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
def test_pairwise_distances_chunked_reduce(global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((400, 4)).astype(global_dtype, copy=False)
    S = pairwise_distances(X)[:, :100]
    S_chunks = pairwise_distances_chunked(X, None, reduce_func=_reduce_func, working_memory=2 ** (-16))
    assert isinstance(S_chunks, GeneratorType)
    S_chunks = list(S_chunks)
    assert len(S_chunks) > 1
    assert S_chunks[0].dtype == X.dtype
    assert_allclose(np.vstack(S_chunks), S, atol=1e-07)