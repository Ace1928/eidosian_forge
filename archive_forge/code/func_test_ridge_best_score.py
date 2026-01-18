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
@pytest.mark.parametrize('ridge, make_dataset', [(RidgeCV(), make_regression), (RidgeClassifierCV(), make_classification)])
@pytest.mark.parametrize('cv', [None, 3])
def test_ridge_best_score(ridge, make_dataset, cv):
    X, y = make_dataset(n_samples=6, random_state=42)
    ridge.set_params(store_cv_values=False, cv=cv)
    ridge.fit(X, y)
    assert hasattr(ridge, 'best_score_')
    assert isinstance(ridge.best_score_, float)