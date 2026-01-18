import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import (
from sklearn.utils.extmath import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('dtype', (np.int32, np.int64, np.float32, np.float64))
def test_randomized_eigsh(dtype):
    """Test that `_randomized_eigsh` returns the appropriate components"""
    rng = np.random.RandomState(42)
    X = np.diag(np.array([1.0, -2.0, 0.0, 3.0], dtype=dtype))
    rand_rot = np.linalg.qr(rng.normal(size=X.shape))[0]
    X = rand_rot @ X @ rand_rot.T
    eigvals, eigvecs = _randomized_eigsh(X, n_components=2, selection='module')
    assert eigvals.shape == (2,)
    assert_array_almost_equal(eigvals, [3.0, -2.0])
    assert eigvecs.shape == (4, 2)
    with pytest.raises(NotImplementedError):
        _randomized_eigsh(X, n_components=2, selection='value')