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
def test_voting_classifier_set_params(global_random_seed):
    clf1 = LogisticRegression(random_state=global_random_seed)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=global_random_seed, max_depth=None)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier([('lr', clf1), ('rf', clf2)], voting='soft', weights=[1, 2]).fit(X_scaled, y)
    eclf2 = VotingClassifier([('lr', clf1), ('nb', clf3)], voting='soft', weights=[1, 2])
    eclf2.set_params(nb=clf2).fit(X_scaled, y)
    assert_array_equal(eclf1.predict(X_scaled), eclf2.predict(X_scaled))
    assert_array_almost_equal(eclf1.predict_proba(X_scaled), eclf2.predict_proba(X_scaled))
    assert eclf2.estimators[0][1].get_params() == clf1.get_params()
    assert eclf2.estimators[1][1].get_params() == clf2.get_params()