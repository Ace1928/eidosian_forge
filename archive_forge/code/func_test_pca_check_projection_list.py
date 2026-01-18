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
def test_pca_check_projection_list(svd_solver):
    X = [[1.0, 0.0], [0.0, 1.0]]
    pca = PCA(n_components=1, svd_solver=svd_solver, random_state=0)
    X_trans = pca.fit_transform(X)
    assert X_trans.shape, (2, 1)
    assert_allclose(X_trans.mean(), 0.0, atol=1e-12)
    assert_allclose(X_trans.std(), 0.71, rtol=0.005)