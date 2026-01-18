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
@pytest.mark.parametrize('scoring', [None, 'accuracy', _accuracy_callable])
def test_ridge_classifier_cv_store_cv_values(scoring):
    x = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 1, 1, -1, -1])
    n_samples = x.shape[0]
    alphas = [0.1, 1.0, 10.0]
    n_alphas = len(alphas)
    scoring_ = make_scorer(scoring) if callable(scoring) else scoring
    r = RidgeClassifierCV(alphas=alphas, cv=None, store_cv_values=True, scoring=scoring_)
    n_targets = 1
    r.fit(x, y)
    assert r.cv_values_.shape == (n_samples, n_targets, n_alphas)
    y = np.array([[1, 1, 1, -1, -1], [1, -1, 1, -1, 1], [-1, -1, 1, -1, -1]]).transpose()
    n_targets = y.shape[1]
    r.fit(x, y)
    assert r.cv_values_.shape == (n_samples, n_targets, n_alphas)