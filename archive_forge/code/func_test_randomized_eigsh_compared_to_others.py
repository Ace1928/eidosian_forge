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
@pytest.mark.parametrize('k', (10, 50, 100, 199, 200))
def test_randomized_eigsh_compared_to_others(k):
    """Check that `_randomized_eigsh` is similar to other `eigsh`

    Tests that for a random PSD matrix, `_randomized_eigsh` provides results
    comparable to LAPACK (scipy.linalg.eigh) and ARPACK
    (scipy.sparse.linalg.eigsh).

    Note: some versions of ARPACK do not support k=n_features.
    """
    n_features = 200
    X = make_sparse_spd_matrix(n_features, random_state=0)
    eigvals, eigvecs = _randomized_eigsh(X, n_components=k, selection='module', n_iter=25, random_state=0)
    eigvals_qr, eigvecs_qr = _randomized_eigsh(X, n_components=k, n_iter=25, n_oversamples=20, random_state=0, power_iteration_normalizer='QR', selection='module')
    eigvals_lapack, eigvecs_lapack = eigh(X, subset_by_index=(n_features - k, n_features - 1))
    indices = eigvals_lapack.argsort()[::-1]
    eigvals_lapack = eigvals_lapack[indices]
    eigvecs_lapack = eigvecs_lapack[:, indices]
    assert eigvals_lapack.shape == (k,)
    assert_array_almost_equal(eigvals, eigvals_lapack, decimal=6)
    assert_array_almost_equal(eigvals_qr, eigvals_lapack, decimal=6)
    assert eigvecs_lapack.shape == (n_features, k)
    dummy_vecs = np.zeros_like(eigvecs).T
    eigvecs, _ = svd_flip(eigvecs, dummy_vecs)
    eigvecs_qr, _ = svd_flip(eigvecs_qr, dummy_vecs)
    eigvecs_lapack, _ = svd_flip(eigvecs_lapack, dummy_vecs)
    assert_array_almost_equal(eigvecs, eigvecs_lapack, decimal=4)
    assert_array_almost_equal(eigvecs_qr, eigvecs_lapack, decimal=6)
    if k < n_features:
        v0 = _init_arpack_v0(n_features, random_state=0)
        eigvals_arpack, eigvecs_arpack = eigsh(X, k, which='LA', tol=0, maxiter=None, v0=v0)
        indices = eigvals_arpack.argsort()[::-1]
        eigvals_arpack = eigvals_arpack[indices]
        assert_array_almost_equal(eigvals_lapack, eigvals_arpack, decimal=10)
        eigvecs_arpack = eigvecs_arpack[:, indices]
        eigvecs_arpack, _ = svd_flip(eigvecs_arpack, dummy_vecs)
        assert_array_almost_equal(eigvecs_arpack, eigvecs_lapack, decimal=8)