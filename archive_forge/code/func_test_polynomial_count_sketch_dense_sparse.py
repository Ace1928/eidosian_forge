import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
@pytest.mark.parametrize('gamma', [0.1, 1.0])
@pytest.mark.parametrize('degree', [1, 2, 3])
@pytest.mark.parametrize('coef0', [0, 2.5])
def test_polynomial_count_sketch_dense_sparse(gamma, degree, coef0, csr_container):
    """Check that PolynomialCountSketch results are the same for dense and sparse
    input.
    """
    ps_dense = PolynomialCountSketch(n_components=500, gamma=gamma, degree=degree, coef0=coef0, random_state=42)
    Xt_dense = ps_dense.fit_transform(X)
    Yt_dense = ps_dense.transform(Y)
    ps_sparse = PolynomialCountSketch(n_components=500, gamma=gamma, degree=degree, coef0=coef0, random_state=42)
    Xt_sparse = ps_sparse.fit_transform(csr_container(X))
    Yt_sparse = ps_sparse.transform(csr_container(Y))
    assert_allclose(Xt_dense, Xt_sparse)
    assert_allclose(Yt_dense, Yt_sparse)