import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_nystroem_poly_kernel_params():
    rnd = np.random.RandomState(37)
    X = rnd.uniform(size=(10, 4))
    K = polynomial_kernel(X, degree=3.1, coef0=0.1)
    nystroem = Nystroem(kernel='polynomial', n_components=X.shape[0], degree=3.1, coef0=0.1)
    X_transformed = nystroem.fit_transform(X)
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)