import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_tol():
    tol = 0.5
    gamma = orthogonal_mp(X, y[:, 0], tol=tol)
    gamma_gram = orthogonal_mp(X, y[:, 0], tol=tol, precompute=True)
    assert np.sum((y[:, 0] - np.dot(X, gamma)) ** 2) <= tol
    assert np.sum((y[:, 0] - np.dot(X, gamma_gram)) ** 2) <= tol