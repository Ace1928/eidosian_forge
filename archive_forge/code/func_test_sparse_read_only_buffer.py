import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('copy_X', (True, False))
def test_sparse_read_only_buffer(copy_X):
    """Test that sparse coordinate descent works for read-only buffers"""
    rng = np.random.RandomState(0)
    clf = ElasticNet(alpha=0.1, copy_X=copy_X, random_state=rng)
    X = sp.random(100, 20, format='csc', random_state=rng)
    X.data = create_memmap_backed_data(X.data)
    y = rng.rand(100)
    clf.fit(X, y)