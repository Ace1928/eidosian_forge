from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import (
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_stacking_final_estimator_attribute_error():
    """Check that we raise the proper AttributeError when the final estimator
    does not implement the `decision_function` method, which is decorated with
    `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    """
    X, y = make_classification(random_state=42)
    estimators = [('lr', LogisticRegression()), ('rf', RandomForestClassifier(n_estimators=2, random_state=42))]
    final_estimator = RandomForestClassifier(n_estimators=2, random_state=42)
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3)
    outer_msg = "This 'StackingClassifier' has no attribute 'decision_function'"
    inner_msg = "'RandomForestClassifier' object has no attribute 'decision_function'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        clf.fit(X, y).decision_function(X)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)