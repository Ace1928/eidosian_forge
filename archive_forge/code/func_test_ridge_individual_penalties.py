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
def test_ridge_individual_penalties():
    rng = np.random.RandomState(42)
    n_samples, n_features, n_targets = (20, 10, 5)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_targets)
    penalties = np.arange(n_targets)
    coef_cholesky = np.array([Ridge(alpha=alpha, solver='cholesky').fit(X, target).coef_ for alpha, target in zip(penalties, y.T)])
    coefs_indiv_pen = [Ridge(alpha=penalties, solver=solver, tol=1e-12).fit(X, y).coef_ for solver in ['svd', 'sparse_cg', 'lsqr', 'cholesky', 'sag', 'saga']]
    for coef_indiv_pen in coefs_indiv_pen:
        assert_array_almost_equal(coef_cholesky, coef_indiv_pen)
    ridge = Ridge(alpha=penalties[:-1])
    err_msg = 'Number of targets and number of penalties do not correspond: 4 != 5'
    with pytest.raises(ValueError, match=err_msg):
        ridge.fit(X, y)