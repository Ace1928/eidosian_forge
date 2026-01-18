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
@pytest.mark.parametrize('ridge, make_dataset', [(RidgeCV(store_cv_values=False), make_regression), (RidgeClassifierCV(store_cv_values=False), make_classification)])
def test_ridge_gcv_cv_values_not_stored(ridge, make_dataset):
    X, y = make_dataset(n_samples=6, random_state=42)
    ridge.fit(X, y)
    assert not hasattr(ridge, 'cv_values_')