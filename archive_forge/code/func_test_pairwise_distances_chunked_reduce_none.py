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
def test_pairwise_distances_chunked_reduce_none(global_dtype):
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 4)).astype(global_dtype, copy=False)
    S_chunks = pairwise_distances_chunked(X, None, reduce_func=lambda dist, start: None, working_memory=2 ** (-16))
    assert isinstance(S_chunks, GeneratorType)
    S_chunks = list(S_chunks)
    assert len(S_chunks) > 1
    assert all((chunk is None for chunk in S_chunks))