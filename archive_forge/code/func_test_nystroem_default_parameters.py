import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_nystroem_default_parameters():
    rnd = np.random.RandomState(42)
    X = rnd.uniform(size=(10, 4))
    nystroem = Nystroem(n_components=10)
    X_transformed = nystroem.fit_transform(X)
    K = rbf_kernel(X, gamma=None)
    K2 = np.dot(X_transformed, X_transformed.T)
    assert_array_almost_equal(K, K2)
    nystroem = Nystroem(kernel='chi2', n_components=10)
    X_transformed = nystroem.fit_transform(X)
    K = chi2_kernel(X, gamma=1)
    K2 = np.dot(X_transformed, X_transformed.T)
    assert_array_almost_equal(K, K2)