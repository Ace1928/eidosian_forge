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
def test_subclass_named_constructors_return_type_is_subclass(pyplot):
    """Check that named constructors return the correct type when subclassed.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/27675
    """
    clf = LogisticRegression().fit(X, y)

    class SubclassOfDisplay(DecisionBoundaryDisplay):
        pass
    curve = SubclassOfDisplay.from_estimator(estimator=clf, X=X)
    assert isinstance(curve, SubclassOfDisplay)