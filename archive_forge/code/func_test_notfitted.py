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
def test_notfitted():
    eclf = VotingClassifier(estimators=[('lr1', LogisticRegression()), ('lr2', LogisticRegression())], voting='soft')
    ereg = VotingRegressor([('dr', DummyRegressor())])
    msg = "This %s instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
    with pytest.raises(NotFittedError, match=msg % 'VotingClassifier'):
        eclf.predict(X)
    with pytest.raises(NotFittedError, match=msg % 'VotingClassifier'):
        eclf.predict_proba(X)
    with pytest.raises(NotFittedError, match=msg % 'VotingClassifier'):
        eclf.transform(X)
    with pytest.raises(NotFittedError, match=msg % 'VotingRegressor'):
        ereg.predict(X_r)
    with pytest.raises(NotFittedError, match=msg % 'VotingRegressor'):
        ereg.transform(X_r)