import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_swapped_regressors():
    gamma = np.zeros(n_features)
    gamma[21] = 1.0
    gamma[0] = 0.5
    new_y = np.dot(X, gamma)
    new_Xy = np.dot(X.T, new_y)
    gamma_hat = orthogonal_mp(X, new_y, n_nonzero_coefs=2)
    gamma_hat_gram = orthogonal_mp_gram(G, new_Xy, n_nonzero_coefs=2)
    assert_array_equal(np.flatnonzero(gamma_hat), [0, 21])
    assert_array_equal(np.flatnonzero(gamma_hat_gram), [0, 21])