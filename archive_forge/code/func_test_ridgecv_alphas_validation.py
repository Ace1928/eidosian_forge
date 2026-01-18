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
@pytest.mark.parametrize('params, err_type, err_msg', [({'alphas': (1, -1, -100)}, ValueError, 'alphas\\[1\\] == -1, must be > 0.0'), ({'alphas': (-0.1, -1.0, -10.0)}, ValueError, 'alphas\\[0\\] == -0.1, must be > 0.0'), ({'alphas': (1, 1.0, '1')}, TypeError, 'alphas\\[2\\] must be an instance of float, not str')])
def test_ridgecv_alphas_validation(Estimator, params, err_type, err_msg):
    """Check the `alphas` validation in RidgeCV and RidgeClassifierCV."""
    n_samples, n_features = (5, 5)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)
    with pytest.raises(err_type, match=err_msg):
        Estimator(**params).fit(X, y)