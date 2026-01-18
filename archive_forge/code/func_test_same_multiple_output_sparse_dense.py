import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_same_multiple_output_sparse_dense(coo_container):
    l = ElasticNet()
    X = [[0, 1, 2, 3, 4], [0, 2, 5, 8, 11], [9, 10, 11, 12, 13], [10, 11, 12, 13, 14]]
    y = [[1, 2, 3, 4, 5], [1, 3, 6, 9, 12], [10, 11, 12, 13, 14], [11, 12, 13, 14, 15]]
    l.fit(X, y)
    sample = np.array([1, 2, 3, 4, 5]).reshape(1, -1)
    predict_dense = l.predict(sample)
    l_sp = ElasticNet()
    X_sp = coo_container(X)
    l_sp.fit(X_sp, y)
    sample_sparse = coo_container(sample)
    predict_sparse = l_sp.predict(sample_sparse)
    assert_array_almost_equal(predict_sparse, predict_dense)