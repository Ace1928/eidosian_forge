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
@pytest.mark.parametrize('svd_solver', ['arpack', 'randomized'])
def test_pca_explained_variance_equivalence_solver(svd_solver):
    rng = np.random.RandomState(0)
    n_samples, n_features = (100, 80)
    X = rng.randn(n_samples, n_features)
    pca_full = PCA(n_components=2, svd_solver='full')
    pca_other = PCA(n_components=2, svd_solver=svd_solver, random_state=0)
    pca_full.fit(X)
    pca_other.fit(X)
    assert_allclose(pca_full.explained_variance_, pca_other.explained_variance_, rtol=0.05)
    assert_allclose(pca_full.explained_variance_ratio_, pca_other.explained_variance_ratio_, rtol=0.05)