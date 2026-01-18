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
def test_randomized_svd_sign_flip_with_transpose():

    def max_loading_is_positive(u, v):
        """
        returns bool tuple indicating if the values maximising np.abs
        are positive across all rows for u and across all columns for v.
        """
        u_based = (np.abs(u).max(axis=0) == u.max(axis=0)).all()
        v_based = (np.abs(v).max(axis=1) == v.max(axis=1)).all()
        return (u_based, v_based)
    mat = np.arange(10 * 8).reshape(10, -1)
    u_flipped, _, v_flipped = randomized_svd(mat, 3, flip_sign=True, random_state=0)
    u_based, v_based = max_loading_is_positive(u_flipped, v_flipped)
    assert u_based
    assert not v_based
    u_flipped_with_transpose, _, v_flipped_with_transpose = randomized_svd(mat, 3, flip_sign=True, transpose=True, random_state=0)
    u_based, v_based = max_loading_is_positive(u_flipped_with_transpose, v_flipped_with_transpose)
    assert u_based
    assert not v_based