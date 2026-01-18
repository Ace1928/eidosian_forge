import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('metric', ['rbf', 'laplacian', 'sigmoid', 'polynomial', 'linear', 'chi2', 'additive_chi2'])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_pairwise_kernels(metric, csr_container):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))
    function = PAIRWISE_KERNEL_FUNCTIONS[metric]
    K1 = pairwise_kernels(X, metric=metric)
    K2 = function(X)
    assert_allclose(K1, K2)
    K1 = pairwise_kernels(X, Y=Y, metric=metric)
    K2 = function(X, Y=Y)
    assert_allclose(K1, K2)
    X_tuples = tuple([tuple([v for v in row]) for row in X])
    Y_tuples = tuple([tuple([v for v in row]) for row in Y])
    K2 = pairwise_kernels(X_tuples, Y_tuples, metric=metric)
    assert_allclose(K1, K2)
    X_sparse = csr_container(X)
    Y_sparse = csr_container(Y)
    if metric in ['chi2', 'additive_chi2']:
        return
    K1 = pairwise_kernels(X_sparse, Y=Y_sparse, metric=metric)
    assert_allclose(K1, K2)