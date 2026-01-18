import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_same_output_sparse_dense_lasso_and_enet_cv(csc_container):
    X, y = make_sparse_data(csc_container, n_samples=40, n_features=10)
    clfs = ElasticNetCV(max_iter=100)
    clfs.fit(X, y)
    clfd = ElasticNetCV(max_iter=100)
    clfd.fit(X.toarray(), y)
    assert_almost_equal(clfs.alpha_, clfd.alpha_, 7)
    assert_almost_equal(clfs.intercept_, clfd.intercept_, 7)
    assert_array_almost_equal(clfs.mse_path_, clfd.mse_path_)
    assert_array_almost_equal(clfs.alphas_, clfd.alphas_)
    clfs = LassoCV(max_iter=100, cv=4)
    clfs.fit(X, y)
    clfd = LassoCV(max_iter=100, cv=4)
    clfd.fit(X.toarray(), y)
    assert_almost_equal(clfs.alpha_, clfd.alpha_, 7)
    assert_almost_equal(clfs.intercept_, clfd.intercept_, 7)
    assert_array_almost_equal(clfs.mse_path_, clfd.mse_path_)
    assert_array_almost_equal(clfs.alphas_, clfd.alphas_)