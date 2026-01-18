import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_rbf_sampler():
    gamma = 10.0
    kernel = rbf_kernel(X, Y, gamma=gamma)
    rbf_transform = RBFSampler(gamma=gamma, n_components=1000, random_state=42)
    X_trans = rbf_transform.fit_transform(X)
    Y_trans = rbf_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) <= 0.01
    np.abs(error, out=error)
    assert np.max(error) <= 0.1
    assert np.mean(error) <= 0.05