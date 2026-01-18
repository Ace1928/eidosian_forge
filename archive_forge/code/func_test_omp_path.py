import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_omp_path():
    path = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=True)
    last = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=False)
    assert path.shape == (n_features, n_targets, 5)
    assert_array_almost_equal(path[:, :, -1], last)
    path = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=True)
    last = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=False)
    assert path.shape == (n_features, n_targets, 5)
    assert_array_almost_equal(path[:, :, -1], last)