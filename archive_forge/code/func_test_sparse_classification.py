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
@pytest.mark.parametrize('sparse_container, params, method', product(CSR_CONTAINERS + CSC_CONTAINERS, [{'max_samples': 0.5, 'max_features': 2, 'bootstrap': True, 'bootstrap_features': True}, {'max_samples': 1.0, 'max_features': 4, 'bootstrap': True, 'bootstrap_features': True}, {'max_features': 2, 'bootstrap': False, 'bootstrap_features': True}, {'max_samples': 0.5, 'bootstrap': True, 'bootstrap_features': False}], ['predict', 'predict_proba', 'predict_log_proba', 'decision_function']))
def test_sparse_classification(sparse_container, params, method):

    class CustomSVC(SVC):
        """SVC variant that records the nature of the training set"""

        def fit(self, X, y):
            super().fit(X, y)
            self.data_type_ = type(X)
            return self
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(scale(iris.data), iris.target, random_state=rng)
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    sparse_classifier = BaggingClassifier(estimator=CustomSVC(kernel='linear', decision_function_shape='ovr'), random_state=1, **params).fit(X_train_sparse, y_train)
    sparse_results = getattr(sparse_classifier, method)(X_test_sparse)
    dense_classifier = BaggingClassifier(estimator=CustomSVC(kernel='linear', decision_function_shape='ovr'), random_state=1, **params).fit(X_train, y_train)
    dense_results = getattr(dense_classifier, method)(X_test)
    assert_array_almost_equal(sparse_results, dense_results)
    sparse_type = type(X_train_sparse)
    types = [i.data_type_ for i in sparse_classifier.estimators_]
    assert all([t == sparse_type for t in types])