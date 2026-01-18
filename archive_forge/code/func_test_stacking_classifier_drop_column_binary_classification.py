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
def test_stacking_classifier_drop_column_binary_classification():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(scale(X), y, stratify=y, random_state=42)
    estimators = [('lr', LogisticRegression()), ('rf', RandomForestClassifier(random_state=42))]
    clf = StackingClassifier(estimators=estimators, cv=3)
    clf.fit(X_train, y_train)
    X_trans = clf.transform(X_test)
    assert X_trans.shape[1] == 2
    estimators = [('lr', LogisticRegression()), ('svc', LinearSVC(dual='auto'))]
    clf.set_params(estimators=estimators)
    clf.fit(X_train, y_train)
    X_trans = clf.transform(X_test)
    assert X_trans.shape[1] == 2