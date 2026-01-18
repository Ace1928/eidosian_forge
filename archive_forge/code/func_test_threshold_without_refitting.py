import re
import warnings
from unittest.mock import Mock
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import make_friedman1
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.utils._testing import (
def test_threshold_without_refitting():
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, random_state=0, tol=None)
    model = SelectFromModel(clf, threshold='0.1 * mean')
    model.fit(data, y)
    X_transform = model.transform(data)
    model.threshold = '1.0 * mean'
    assert X_transform.shape[1] > model.transform(data).shape[1]