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
@pytest.mark.parametrize('n', [50, 100, 300])
@pytest.mark.parametrize('m', [50, 100, 300])
@pytest.mark.parametrize('k', [10, 20, 50])
@pytest.mark.parametrize('seed', range(5))
def test_randomized_svd_lapack_driver(n, m, k, seed):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, m)
    u1, s1, vt1 = randomized_svd(X, k, svd_lapack_driver='gesdd', random_state=0)
    u2, s2, vt2 = randomized_svd(X, k, svd_lapack_driver='gesvd', random_state=0)
    assert u1.shape == u2.shape
    assert_allclose(u1, u2, atol=0, rtol=0.001)
    assert s1.shape == s2.shape
    assert_allclose(s1, s2, atol=0, rtol=0.001)
    assert vt1.shape == vt2.shape
    assert_allclose(vt1, vt2, atol=0, rtol=0.001)