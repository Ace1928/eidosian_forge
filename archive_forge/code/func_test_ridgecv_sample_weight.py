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
def test_ridgecv_sample_weight():
    rng = np.random.RandomState(0)
    alphas = (0.1, 1.0, 10.0)
    for n_samples, n_features in ((6, 5), (5, 10)):
        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)
        sample_weight = 1.0 + rng.rand(n_samples)
        cv = KFold(5)
        ridgecv = RidgeCV(alphas=alphas, cv=cv)
        ridgecv.fit(X, y, sample_weight=sample_weight)
        parameters = {'alpha': alphas}
        gs = GridSearchCV(Ridge(), parameters, cv=cv)
        gs.fit(X, y, sample_weight=sample_weight)
        assert ridgecv.alpha_ == gs.best_estimator_.alpha
        assert_array_almost_equal(ridgecv.coef_, gs.best_estimator_.coef_)