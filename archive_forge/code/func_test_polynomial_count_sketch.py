import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('gamma', [0.1, 1, 2.5])
@pytest.mark.parametrize('degree, n_components', [(1, 500), (2, 500), (3, 5000)])
@pytest.mark.parametrize('coef0', [0, 2.5])
def test_polynomial_count_sketch(gamma, degree, coef0, n_components):
    kernel = polynomial_kernel(X, Y, gamma=gamma, degree=degree, coef0=coef0)
    ps_transform = PolynomialCountSketch(n_components=n_components, gamma=gamma, coef0=coef0, degree=degree, random_state=42)
    X_trans = ps_transform.fit_transform(X)
    Y_trans = ps_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) <= 0.05
    np.abs(error, out=error)
    assert np.max(error) <= 0.1
    assert np.mean(error) <= 0.05