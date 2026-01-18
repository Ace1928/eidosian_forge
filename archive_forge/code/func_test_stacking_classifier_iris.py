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
@pytest.mark.parametrize('cv', [3, StratifiedKFold(n_splits=3, shuffle=True, random_state=42)])
@pytest.mark.parametrize('final_estimator', [None, RandomForestClassifier(random_state=42)])
@pytest.mark.parametrize('passthrough', [False, True])
def test_stacking_classifier_iris(cv, final_estimator, passthrough):
    X_train, X_test, y_train, y_test = train_test_split(scale(X_iris), y_iris, stratify=y_iris, random_state=42)
    estimators = [('lr', LogisticRegression()), ('svc', LinearSVC(dual='auto'))]
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=cv, passthrough=passthrough)
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    clf.predict_proba(X_test)
    assert clf.score(X_test, y_test) > 0.8
    X_trans = clf.transform(X_test)
    expected_column_count = 10 if passthrough else 6
    assert X_trans.shape[1] == expected_column_count
    if passthrough:
        assert_allclose(X_test, X_trans[:, -4:])
    clf.set_params(lr='drop')
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    clf.predict_proba(X_test)
    if final_estimator is None:
        clf.decision_function(X_test)
    X_trans = clf.transform(X_test)
    expected_column_count_drop = 7 if passthrough else 3
    assert X_trans.shape[1] == expected_column_count_drop
    if passthrough:
        assert_allclose(X_test, X_trans[:, -4:])