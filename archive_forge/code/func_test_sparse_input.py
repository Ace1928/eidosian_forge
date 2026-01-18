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
@skip_if_32bit
@pytest.mark.parametrize('EstimatorClass', (GradientBoostingClassifier, GradientBoostingRegressor))
@pytest.mark.parametrize('sparse_container', COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS)
def test_sparse_input(EstimatorClass, sparse_container):
    y, X = datasets.make_multilabel_classification(random_state=0, n_samples=50, n_features=1, n_classes=20)
    y = y[:, 0]
    X_sparse = sparse_container(X)
    dense = EstimatorClass(n_estimators=10, random_state=0, max_depth=2, min_impurity_decrease=1e-07).fit(X, y)
    sparse = EstimatorClass(n_estimators=10, random_state=0, max_depth=2, min_impurity_decrease=1e-07).fit(X_sparse, y)
    assert_array_almost_equal(sparse.apply(X), dense.apply(X))
    assert_array_almost_equal(sparse.predict(X), dense.predict(X))
    assert_array_almost_equal(sparse.feature_importances_, dense.feature_importances_)
    assert_array_almost_equal(sparse.predict(X_sparse), dense.predict(X))
    assert_array_almost_equal(dense.predict(X_sparse), sparse.predict(X))
    if issubclass(EstimatorClass, GradientBoostingClassifier):
        assert_array_almost_equal(sparse.predict_proba(X), dense.predict_proba(X))
        assert_array_almost_equal(sparse.predict_log_proba(X), dense.predict_log_proba(X))
        assert_array_almost_equal(sparse.decision_function(X_sparse), sparse.decision_function(X))
        assert_array_almost_equal(dense.decision_function(X_sparse), sparse.decision_function(X))
        for res_sparse, res in zip(sparse.staged_decision_function(X_sparse), sparse.staged_decision_function(X)):
            assert_array_almost_equal(res_sparse, res)