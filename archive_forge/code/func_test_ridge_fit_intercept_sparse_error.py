import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('solver', ['saga', 'svd', 'cholesky'])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_ridge_fit_intercept_sparse_error(solver, csr_container):
    X, y = _make_sparse_offset_regression(n_features=20, random_state=0)
    X_csr = csr_container(X)
    sparse_ridge = Ridge(solver=solver)
    err_msg = "solver='{}' does not support".format(solver)
    with pytest.raises(ValueError, match=err_msg):
        sparse_ridge.fit(X_csr, y)