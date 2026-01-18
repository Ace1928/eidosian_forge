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
def test_set_estimator_drop():
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=123)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3)], voting='hard', weights=[1, 0, 0.5]).fit(X, y)
    eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('nb', clf3)], voting='hard', weights=[1, 1, 0.5])
    eclf2.set_params(rf='drop').fit(X, y)
    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
    assert dict(eclf2.estimators)['rf'] == 'drop'
    assert len(eclf2.estimators_) == 2
    assert all((isinstance(est, (LogisticRegression, GaussianNB)) for est in eclf2.estimators_))
    assert eclf2.get_params()['rf'] == 'drop'
    eclf1.set_params(voting='soft').fit(X, y)
    eclf2.set_params(voting='soft').fit(X, y)
    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
    assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))
    msg = 'All estimators are dropped. At least one is required'
    with pytest.raises(ValueError, match=msg):
        eclf2.set_params(lr='drop', rf='drop', nb='drop').fit(X, y)
    X1 = np.array([[1], [2]])
    y1 = np.array([1, 2])
    eclf1 = VotingClassifier(estimators=[('rf', clf2), ('nb', clf3)], voting='soft', weights=[0, 0.5], flatten_transform=False).fit(X1, y1)
    eclf2 = VotingClassifier(estimators=[('rf', clf2), ('nb', clf3)], voting='soft', weights=[1, 0.5], flatten_transform=False)
    eclf2.set_params(rf='drop').fit(X1, y1)
    assert_array_almost_equal(eclf1.transform(X1), np.array([[[0.7, 0.3], [0.3, 0.7]], [[1.0, 0.0], [0.0, 1.0]]]))
    assert_array_almost_equal(eclf2.transform(X1), np.array([[[1.0, 0.0], [0.0, 1.0]]]))
    eclf1.set_params(voting='hard')
    eclf2.set_params(voting='hard')
    assert_array_equal(eclf1.transform(X1), np.array([[0, 0], [1, 1]]))
    assert_array_equal(eclf2.transform(X1), np.array([[0], [1]]))