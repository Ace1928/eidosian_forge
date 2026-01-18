import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import _safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_regression_synthetic(global_random_seed):
    random_state = check_random_state(global_random_seed)
    regression_params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.1, 'loss': 'squared_error', 'random_state': global_random_seed}
    X, y = datasets.make_friedman1(n_samples=1200, random_state=random_state, noise=1.0)
    X_train, y_train = (X[:200], y[:200])
    X_test, y_test = (X[200:], y[200:])
    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    assert mse < 6.5
    X, y = datasets.make_friedman2(n_samples=1200, random_state=random_state)
    X_train, y_train = (X[:200], y[:200])
    X_test, y_test = (X[200:], y[200:])
    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    assert mse < 2500.0
    X, y = datasets.make_friedman3(n_samples=1200, random_state=random_state)
    X_train, y_train = (X[:200], y[:200])
    X_test, y_test = (X[200:], y[200:])
    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    assert mse < 0.025