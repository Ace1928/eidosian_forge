import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_orthogonal_mp_gram_readonly():
    idx, = gamma[:, 0].nonzero()
    G_readonly = G.copy()
    G_readonly.setflags(write=False)
    Xy_readonly = Xy.copy()
    Xy_readonly.setflags(write=False)
    gamma_gram = orthogonal_mp_gram(G_readonly, Xy_readonly[:, 0], n_nonzero_coefs=5, copy_Gram=False, copy_Xy=False)
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)