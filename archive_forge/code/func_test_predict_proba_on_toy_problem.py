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
def test_predict_proba_on_toy_problem():
    """Calculate predicted probabilities on toy dataset."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])
    clf1_res = np.array([[0.59790391, 0.40209609], [0.57622162, 0.42377838], [0.50728456, 0.49271544], [0.40241774, 0.59758226]])
    clf2_res = np.array([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]])
    clf3_res = np.array([[0.9985082, 0.0014918], [0.99845843, 0.00154157], [0.0, 1.0], [0.0, 1.0]])
    t00 = (2 * clf1_res[0][0] + clf2_res[0][0] + clf3_res[0][0]) / 4
    t11 = (2 * clf1_res[1][1] + clf2_res[1][1] + clf3_res[1][1]) / 4
    t21 = (2 * clf1_res[2][1] + clf2_res[2][1] + clf3_res[2][1]) / 4
    t31 = (2 * clf1_res[3][1] + clf2_res[3][1] + clf3_res[3][1]) / 4
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[2, 1, 1])
    eclf_res = eclf.fit(X, y).predict_proba(X)
    assert_almost_equal(t00, eclf_res[0][0], decimal=1)
    assert_almost_equal(t11, eclf_res[1][1], decimal=1)
    assert_almost_equal(t21, eclf_res[2][1], decimal=1)
    assert_almost_equal(t31, eclf_res[3][1], decimal=1)
    inner_msg = "predict_proba is not available when voting='hard'"
    outer_msg = "'VotingClassifier' has no attribute 'predict_proba'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
        eclf.fit(X, y).predict_proba(X)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)