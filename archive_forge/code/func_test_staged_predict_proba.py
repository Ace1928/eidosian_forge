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
def test_staged_predict_proba():
    X, y = datasets.make_hastie_10_2(n_samples=1200, random_state=1)
    X_train, y_train = (X[:200], y[:200])
    X_test, y_test = (X[200:], y[200:])
    clf = GradientBoostingClassifier(n_estimators=20)
    with pytest.raises(NotFittedError):
        np.fromiter(clf.staged_predict_proba(X_test), dtype=np.float64)
    clf.fit(X_train, y_train)
    for y_pred in clf.staged_predict(X_test):
        assert y_test.shape == y_pred.shape
    assert_array_equal(clf.predict(X_test), y_pred)
    for staged_proba in clf.staged_predict_proba(X_test):
        assert y_test.shape[0] == staged_proba.shape[0]
        assert 2 == staged_proba.shape[1]
    assert_array_almost_equal(clf.predict_proba(X_test), staged_proba)