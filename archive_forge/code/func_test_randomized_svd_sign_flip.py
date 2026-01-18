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
def test_randomized_svd_sign_flip():
    a = np.array([[2.0, 0.0], [0.0, 1.0]])
    u1, s1, v1 = randomized_svd(a, 2, flip_sign=True, random_state=41)
    for seed in range(10):
        u2, s2, v2 = randomized_svd(a, 2, flip_sign=True, random_state=seed)
        assert_almost_equal(u1, u2)
        assert_almost_equal(v1, v2)
        assert_almost_equal(np.dot(u2 * s2, v2), a)
        assert_almost_equal(np.dot(u2.T, u2), np.eye(2))
        assert_almost_equal(np.dot(v2.T, v2), np.eye(2))