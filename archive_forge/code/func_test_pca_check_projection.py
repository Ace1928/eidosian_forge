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
def test_pca_check_projection(svd_solver):
    rng = np.random.RandomState(0)
    n, p = (100, 3)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])
    Yt = PCA(n_components=2, svd_solver=svd_solver).fit(X).transform(Xt)
    Yt /= np.sqrt((Yt ** 2).sum())
    assert_allclose(np.abs(Yt[0][0]), 1.0, rtol=0.005)