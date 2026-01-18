import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_multilabel_classification
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
def test_predict_on_toy_problem(global_random_seed):
    """Manually check predicted class labels for toy dataset."""
    clf1 = LogisticRegression(random_state=global_random_seed)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed)
    clf3 = GaussianNB()
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2], [2.1, 1.4], [3.1, 2.3]])
    y = np.array([1, 1, 1, 2, 2, 2])
    assert_array_equal(clf1.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
    assert_array_equal(clf2.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
    assert_array_equal(clf3.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard', weights=[1, 1, 1])
    assert_array_equal(eclf.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1, 1, 1])
    assert_array_equal(eclf.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])