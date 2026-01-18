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
def test_stacking_regressor_drop_estimator():
    X_train, X_test, y_train, _ = train_test_split(scale(X_diabetes), y_diabetes, random_state=42)
    estimators = [('lr', 'drop'), ('svr', LinearSVR(dual='auto', random_state=0))]
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    reg = StackingRegressor(estimators=[('svr', LinearSVR(dual='auto', random_state=0))], final_estimator=rf, cv=5)
    reg_drop = StackingRegressor(estimators=estimators, final_estimator=rf, cv=5)
    reg.fit(X_train, y_train)
    reg_drop.fit(X_train, y_train)
    assert_allclose(reg.predict(X_test), reg_drop.predict(X_test))
    assert_allclose(reg.transform(X_test), reg_drop.transform(X_test))