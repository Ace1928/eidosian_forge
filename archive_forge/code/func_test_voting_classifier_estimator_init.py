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
@pytest.mark.parametrize('params, err_msg', [({'estimators': []}, "Invalid 'estimators' attribute, 'estimators' should be a non-empty list"), ({'estimators': [('lr', LogisticRegression())], 'weights': [1, 2]}, 'Number of `estimators` and weights must be equal')])
def test_voting_classifier_estimator_init(params, err_msg):
    ensemble = VotingClassifier(**params)
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)