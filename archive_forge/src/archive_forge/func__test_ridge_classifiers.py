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
def _test_ridge_classifiers(sparse_container):
    n_classes = np.unique(y_iris).shape[0]
    n_features = X_iris.shape[1]
    X = X_iris if sparse_container is None else sparse_container(X_iris)
    for reg in (RidgeClassifier(), RidgeClassifierCV()):
        reg.fit(X, y_iris)
        assert reg.coef_.shape == (n_classes, n_features)
        y_pred = reg.predict(X)
        assert np.mean(y_iris == y_pred) > 0.79
    cv = KFold(5)
    reg = RidgeClassifierCV(cv=cv)
    reg.fit(X, y_iris)
    y_pred = reg.predict(X)
    assert np.mean(y_iris == y_pred) >= 0.8