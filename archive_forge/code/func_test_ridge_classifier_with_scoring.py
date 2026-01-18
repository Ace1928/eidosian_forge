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
@pytest.mark.parametrize('cv', [None, KFold(5)])
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
def test_ridge_classifier_with_scoring(sparse_container, scoring, cv):
    X = X_iris if sparse_container is None else sparse_container(X_iris)
    scoring_ = make_scorer(scoring) if callable(scoring) else scoring
    clf = RidgeClassifierCV(scoring=scoring_, cv=cv)
    clf.fit(X, y_iris).predict(X)