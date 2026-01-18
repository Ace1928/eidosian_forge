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
@pytest.mark.parametrize('n,rank', [(10, 7), (100, 10), (100, 80), (500, 10), (500, 250), (500, 400)])
def test_randomized_eigsh_reconst_low_rank(n, rank):
    """Check that randomized_eigsh is able to reconstruct a low rank psd matrix

    Tests that the decomposition provided by `_randomized_eigsh` leads to
    orthonormal eigenvectors, and that a low rank PSD matrix can be effectively
    reconstructed with good accuracy using it.
    """
    assert rank < n
    rng = np.random.RandomState(69)
    X = rng.randn(n, rank)
    A = X @ X.T
    S, V = _randomized_eigsh(A, n_components=rank, random_state=rng)
    assert_array_almost_equal(np.linalg.norm(V, axis=0), np.ones(S.shape))
    assert_array_almost_equal(V.T @ V, np.diag(np.ones(S.shape)))
    A_reconstruct = V @ np.diag(S) @ V.T
    assert_array_almost_equal(A_reconstruct, A, decimal=6)