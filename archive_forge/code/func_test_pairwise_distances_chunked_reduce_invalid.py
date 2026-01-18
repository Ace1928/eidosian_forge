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
@pytest.mark.parametrize(('bad_reduce', 'err_type', 'message'), [(lambda D, s: np.concatenate([D, D[-1:]]), ValueError, 'length 11\\..* input: 10\\.'), (lambda D, s: (D, np.concatenate([D, D[-1:]])), ValueError, 'length \\(10, 11\\)\\..* input: 10\\.'), (lambda D, s: (D[:9], D), ValueError, 'length \\(9, 10\\)\\..* input: 10\\.'), (lambda D, s: 7, TypeError, 'returned 7\\. Expected sequence\\(s\\) of length 10\\.'), (lambda D, s: (7, 8), TypeError, 'returned \\(7, 8\\)\\. Expected sequence\\(s\\) of length 10\\.'), (lambda D, s: (np.arange(10), 9), TypeError, ', 9\\)\\. Expected sequence\\(s\\) of length 10\\.')])
def test_pairwise_distances_chunked_reduce_invalid(global_dtype, bad_reduce, err_type, message):
    X = np.arange(10).reshape(-1, 1).astype(global_dtype, copy=False)
    S_chunks = pairwise_distances_chunked(X, None, reduce_func=bad_reduce, working_memory=64)
    with pytest.raises(err_type, match=message):
        next(S_chunks)