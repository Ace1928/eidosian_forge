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
@pytest.mark.parametrize('svd_solver', PCA_SOLVERS)
@pytest.mark.parametrize('n_components', range(1, iris.data.shape[1]))
def test_pca(svd_solver, n_components):
    X = iris.data
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    X_r = pca.fit(X).transform(X)
    assert X_r.shape[1] == n_components
    X_r2 = pca.fit_transform(X)
    assert_allclose(X_r, X_r2)
    X_r = pca.transform(X)
    assert_allclose(X_r, X_r2)
    cov = pca.get_covariance()
    precision = pca.get_precision()
    assert_allclose(np.dot(cov, precision), np.eye(X.shape[1]), atol=1e-12)