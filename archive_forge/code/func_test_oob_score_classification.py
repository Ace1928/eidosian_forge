from itertools import cycle, product
import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import load_diabetes, load_iris, make_hastie_10_2
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, scale
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_oob_score_classification():
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=rng)
    for estimator in [DecisionTreeClassifier(), SVC()]:
        clf = BaggingClassifier(estimator=estimator, n_estimators=100, bootstrap=True, oob_score=True, random_state=rng).fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        assert abs(test_score - clf.oob_score_) < 0.1
        warn_msg = 'Some inputs do not have OOB scores. This probably means too few estimators were used to compute any reliable oob estimates.'
        with pytest.warns(UserWarning, match=warn_msg):
            clf = BaggingClassifier(estimator=estimator, n_estimators=1, bootstrap=True, oob_score=True, random_state=rng)
            clf.fit(X_train, y_train)