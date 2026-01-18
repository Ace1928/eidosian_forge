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
@pytest.mark.parametrize('n_samples, n_features', [(3, 4), (4, 3)])
def test_svd_flip_max_abs_cols(n_samples, n_features, global_random_seed):
    rs = np.random.RandomState(global_random_seed)
    X = rs.randn(n_samples, n_features)
    U, _, Vt = linalg.svd(X, full_matrices=False)
    U1, _ = svd_flip(U, Vt, u_based_decision=True)
    max_abs_U1_row_idx_for_col = np.argmax(np.abs(U1), axis=0)
    assert (U1[max_abs_U1_row_idx_for_col, np.arange(U1.shape[1])] >= 0).all()
    _, V2 = svd_flip(U, Vt, u_based_decision=False)
    max_abs_V2_col_idx_for_row = np.argmax(np.abs(V2), axis=1)
    assert (V2[np.arange(V2.shape[0]), max_abs_V2_col_idx_for_row] >= 0).all()