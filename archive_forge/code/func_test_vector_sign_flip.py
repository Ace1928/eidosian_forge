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
def test_vector_sign_flip():
    data = np.random.RandomState(36).randn(5, 5)
    max_abs_rows = np.argmax(np.abs(data), axis=1)
    data_flipped = _deterministic_vector_sign_flip(data)
    max_rows = np.argmax(data_flipped, axis=1)
    assert_array_equal(max_abs_rows, max_rows)
    signs = np.sign(data[range(data.shape[0]), max_abs_rows])
    assert_array_equal(data, data_flipped * signs[:, np.newaxis])