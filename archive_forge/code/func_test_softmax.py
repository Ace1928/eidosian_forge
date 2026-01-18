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
def test_softmax():
    rng = np.random.RandomState(0)
    X = rng.randn(3, 5)
    exp_X = np.exp(X)
    sum_exp_X = np.sum(exp_X, axis=1).reshape((-1, 1))
    assert_array_almost_equal(softmax(X), exp_X / sum_exp_X)