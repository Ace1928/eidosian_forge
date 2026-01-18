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
@pytest.mark.parametrize('X, y, voter', [(X, y, VotingClassifier([('lr', LogisticRegression()), ('rf', RandomForestClassifier(n_estimators=5))])), (X_r, y_r, VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor(n_estimators=5))]))])
def test_none_estimator_with_weights(X, y, voter):
    voter = clone(voter)
    X_scaled = StandardScaler().fit_transform(X)
    voter.fit(X_scaled, y, sample_weight=np.ones(y.shape))
    voter.set_params(lr='drop')
    voter.fit(X_scaled, y, sample_weight=np.ones(y.shape))
    y_pred = voter.predict(X_scaled)
    assert y_pred.shape == y.shape