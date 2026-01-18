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
@pytest.mark.parametrize('Estimator', [RidgeCV, RidgeClassifierCV])
def test_ridgecv_alphas_conversion(Estimator):
    rng = np.random.RandomState(0)
    alphas = (0.1, 1.0, 10.0)
    n_samples, n_features = (5, 5)
    if Estimator is RidgeCV:
        y = rng.randn(n_samples)
    else:
        y = rng.randint(0, 2, n_samples)
    X = rng.randn(n_samples, n_features)
    ridge_est = Estimator(alphas=alphas)
    assert ridge_est.alphas is alphas, f'`alphas` was mutated in `{Estimator.__name__}.__init__`'
    ridge_est.fit(X, y)
    assert_array_equal(ridge_est.alphas, np.asarray(alphas))