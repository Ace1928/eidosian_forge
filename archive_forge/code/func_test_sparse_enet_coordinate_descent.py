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
def test_sparse_enet_coordinate_descent(csc_container):
    """Test that a warning is issued if model does not converge"""
    clf = Lasso(max_iter=2)
    n_samples = 5
    n_features = 2
    X = csc_container((n_samples, n_features)) * 1e+50
    y = np.ones(n_samples)
    warning_message = 'Objective did not converge. You might want to increase the number of iterations.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, y)