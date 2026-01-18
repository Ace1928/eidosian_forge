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
@pytest.mark.parametrize('data', [np.array([[0, 1, 0], [1, 0, 0]]), np.array([[0, 1, 0], [1, 0, 0]]).T])
@pytest.mark.parametrize('svd_solver, n_components, err_msg', [('arpack', 0, 'must be between 1 and min\\(n_samples, n_features\\)'), ('randomized', 0, 'must be between 1 and min\\(n_samples, n_features\\)'), ('arpack', 2, 'must be strictly less than min'), ('auto', 3, "n_components=3 must be between 0 and min\\(n_samples, n_features\\)=2 with svd_solver='full'")])
def test_pca_validation(svd_solver, data, n_components, err_msg):
    smallest_d = 2
    pca_fitted = PCA(n_components, svd_solver=svd_solver)
    with pytest.raises(ValueError, match=err_msg):
        pca_fitted.fit(data)
    if svd_solver == 'arpack':
        n_components = smallest_d
        err_msg = "n_components={}L? must be strictly less than min\\(n_samples, n_features\\)={}L? with svd_solver='arpack'".format(n_components, smallest_d)
        with pytest.raises(ValueError, match=err_msg):
            PCA(n_components, svd_solver=svd_solver).fit(data)