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
def test_ridgecv_alphas_scalar(Estimator):
    """Check the case when `alphas` is a scalar.
    This case was supported in the past when `alphas` where converted
    into array in `__init__`.
    We add this test to ensure backward compatibility.
    """
    n_samples, n_features = (5, 5)
    X = rng.randn(n_samples, n_features)
    if Estimator is RidgeCV:
        y = rng.randn(n_samples)
    else:
        y = rng.randint(0, 2, n_samples)
    Estimator(alphas=1).fit(X, y)