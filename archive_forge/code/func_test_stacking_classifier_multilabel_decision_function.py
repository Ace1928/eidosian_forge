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
def test_stacking_classifier_multilabel_decision_function():
    """Check the behaviour for the multilabel classification case and the
    `decision_function` stacking method. Only `RidgeClassifier` supports this
    case.
    """
    X_train, X_test, y_train, y_test = train_test_split(X_multilabel, y_multilabel, stratify=y_multilabel, random_state=42)
    n_outputs = 3
    estimators = [('est', RidgeClassifier())]
    stacker = StackingClassifier(estimators=estimators, final_estimator=KNeighborsClassifier(), stack_method='decision_function').fit(X_train, y_train)
    X_trans = stacker.transform(X_test)
    assert X_trans.shape == (X_test.shape[0], n_outputs)
    y_pred = stacker.predict(X_test)
    assert y_pred.shape == y_test.shape