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
def test_sample_weight_kwargs():
    """Check that VotingClassifier passes sample_weight as kwargs"""

    class MockClassifier(ClassifierMixin, BaseEstimator):
        """Mock Classifier to check that sample_weight is received as kwargs"""

        def fit(self, X, y, *args, **sample_weight):
            assert 'sample_weight' in sample_weight
    clf = MockClassifier()
    eclf = VotingClassifier(estimators=[('mock', clf)], voting='soft')
    eclf.fit(X, y, sample_weight=np.ones((len(y),)))