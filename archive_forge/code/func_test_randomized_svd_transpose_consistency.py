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
def test_randomized_svd_transpose_consistency():
    n_samples = 100
    n_features = 500
    rank = 4
    k = 10
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=rank, tail_strength=0.5, random_state=0)
    assert X.shape == (n_samples, n_features)
    U1, s1, V1 = randomized_svd(X, k, n_iter=3, transpose=False, random_state=0)
    U2, s2, V2 = randomized_svd(X, k, n_iter=3, transpose=True, random_state=0)
    U3, s3, V3 = randomized_svd(X, k, n_iter=3, transpose='auto', random_state=0)
    U4, s4, V4 = linalg.svd(X, full_matrices=False)
    assert_almost_equal(s1, s4[:k], decimal=3)
    assert_almost_equal(s2, s4[:k], decimal=3)
    assert_almost_equal(s3, s4[:k], decimal=3)
    assert_almost_equal(np.dot(U1, V1), np.dot(U4[:, :k], V4[:k, :]), decimal=2)
    assert_almost_equal(np.dot(U2, V2), np.dot(U4[:, :k], V4[:k, :]), decimal=2)
    assert_almost_equal(s2, s3)