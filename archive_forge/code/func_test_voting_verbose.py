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
@pytest.mark.parametrize('estimator', [VotingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor(random_state=123))], verbose=True), VotingClassifier(estimators=[('lr', LogisticRegression(random_state=123)), ('rf', RandomForestClassifier(random_state=123))], verbose=True)])
def test_voting_verbose(estimator, capsys):
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])
    pattern = '\\[Voting\\].*\\(1 of 2\\) Processing lr, total=.*\\n\\[Voting\\].*\\(2 of 2\\) Processing rf, total=.*\\n$'
    estimator.fit(X, y)
    assert re.match(pattern, capsys.readouterr()[0])