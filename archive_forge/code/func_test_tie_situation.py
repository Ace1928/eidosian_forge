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
def test_tie_situation():
    """Check voting classifier selects smaller class label in tie situation."""
    clf1 = LogisticRegression(random_state=123, solver='liblinear')
    clf2 = RandomForestClassifier(random_state=123)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='hard')
    assert clf1.fit(X, y).predict(X)[73] == 2
    assert clf2.fit(X, y).predict(X)[73] == 1
    assert eclf.fit(X, y).predict(X)[73] == 1