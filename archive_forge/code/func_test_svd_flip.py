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
def test_svd_flip():
    rs = np.random.RandomState(1999)
    n_samples = 20
    n_features = 10
    X = rs.randn(n_samples, n_features)
    U, S, Vt = linalg.svd(X, full_matrices=False)
    U1, V1 = svd_flip(U, Vt, u_based_decision=False)
    assert_almost_equal(np.dot(U1 * S, V1), X, decimal=6)
    XT = X.T
    U, S, Vt = linalg.svd(XT, full_matrices=False)
    U2, V2 = svd_flip(U, Vt, u_based_decision=True)
    assert_almost_equal(np.dot(U2 * S, V2), XT, decimal=6)
    U_flip1, V_flip1 = svd_flip(U, Vt, u_based_decision=True)
    assert_almost_equal(np.dot(U_flip1 * S, V_flip1), XT, decimal=6)
    U_flip2, V_flip2 = svd_flip(U, Vt, u_based_decision=False)
    assert_almost_equal(np.dot(U_flip2 * S, V_flip2), XT, decimal=6)