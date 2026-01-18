import warnings
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
@pytest.mark.parametrize('response_method', ['predict_proba', 'decision_function', ['predict_proba', 'predict']])
def test_regressor_unsupported_response(pyplot, response_method):
    """Check that we can display the decision boundary for a regressor."""
    X, y = load_diabetes(return_X_y=True)
    X = X[:, :2]
    tree = DecisionTreeRegressor().fit(X, y)
    err_msg = 'should either be a classifier to be used with response_method'
    with pytest.raises(ValueError, match=err_msg):
        DecisionBoundaryDisplay.from_estimator(tree, X, response_method=response_method)