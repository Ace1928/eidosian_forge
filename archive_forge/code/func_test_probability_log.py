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
def test_probability_log(global_random_seed):
    clf = GradientBoostingClassifier(n_estimators=100, random_state=global_random_seed)
    with pytest.raises(ValueError):
        clf.predict_proba(T)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    y_proba = clf.predict_proba(T)
    assert np.all(y_proba >= 0.0)
    assert np.all(y_proba <= 1.0)
    y_pred = clf.classes_.take(y_proba.argmax(axis=1), axis=0)
    assert_array_equal(y_pred, true_result)