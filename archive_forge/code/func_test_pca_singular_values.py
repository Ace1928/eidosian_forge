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
def test_pca_singular_values(svd_solver):
    rng = np.random.RandomState(0)
    n_samples, n_features = (100, 80)
    X = rng.randn(n_samples, n_features)
    pca = PCA(n_components=2, svd_solver=svd_solver, random_state=rng)
    X_trans = pca.fit_transform(X)
    assert_allclose(np.sum(pca.singular_values_ ** 2), np.linalg.norm(X_trans, 'fro') ** 2)
    assert_allclose(pca.singular_values_, np.sqrt(np.sum(X_trans ** 2, axis=0)))
    n_samples, n_features = (100, 110)
    X = rng.randn(n_samples, n_features)
    pca = PCA(n_components=3, svd_solver=svd_solver, random_state=rng)
    X_trans = pca.fit_transform(X)
    X_trans /= np.sqrt(np.sum(X_trans ** 2, axis=0))
    X_trans[:, 0] *= 3.142
    X_trans[:, 1] *= 2.718
    X_hat = np.dot(X_trans, pca.components_)
    pca.fit(X_hat)
    assert_allclose(pca.singular_values_, [3.142, 2.718, 1.0])