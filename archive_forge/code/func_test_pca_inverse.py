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
@pytest.mark.parametrize('svd_solver', ['full', 'arpack', 'randomized'])
@pytest.mark.parametrize('whiten', [False, True])
def test_pca_inverse(svd_solver, whiten):
    rng = np.random.RandomState(0)
    n, p = (50, 3)
    X = rng.randn(n, p)
    X[:, 1] *= 1e-05
    X += [5, 4, 3]
    pca = PCA(n_components=2, svd_solver=svd_solver, whiten=whiten).fit(X)
    Y = pca.transform(X)
    Y_inverse = pca.inverse_transform(Y)
    assert_allclose(X, Y_inverse, rtol=5e-06)