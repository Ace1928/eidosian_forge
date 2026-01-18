import re
import warnings
import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_array_equal
from sklearn import config_context, datasets
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension, _infer_dimension
from sklearn.utils._array_api import (
from sklearn.utils._array_api import device as array_device
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + CSC_CONTAINERS)
def test_pca_sparse_fit_transform(global_random_seed, sparse_container):
    random_state = np.random.default_rng(global_random_seed)
    X = sparse_container(sp.sparse.random(SPARSE_M, SPARSE_N, random_state=random_state, density=0.01))
    X2 = sparse_container(sp.sparse.random(SPARSE_M, SPARSE_N, random_state=random_state, density=0.01))
    pca_fit = PCA(n_components=10, svd_solver='arpack', random_state=global_random_seed)
    pca_fit_transform = PCA(n_components=10, svd_solver='arpack', random_state=global_random_seed)
    pca_fit.fit(X)
    transformed_X = pca_fit_transform.fit_transform(X)
    _check_fitted_pca_close(pca_fit, pca_fit_transform, rtol=1e-10)
    assert_allclose(transformed_X, pca_fit_transform.transform(X), rtol=2e-09)
    assert_allclose(transformed_X, pca_fit.transform(X), rtol=2e-09)
    assert_allclose(pca_fit.transform(X2), pca_fit_transform.transform(X2), rtol=2e-09)