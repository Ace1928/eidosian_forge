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
def test_prefit_max_features():
    """Check the interaction between `prefit` and `max_features`."""
    estimator = RandomForestClassifier(n_estimators=5, random_state=0)
    estimator.fit(data, y)
    model = SelectFromModel(estimator, prefit=True, max_features=lambda X: X.shape[1])
    err_msg = 'When `prefit=True` and `max_features` is a callable, call `fit` before calling `transform`.'
    with pytest.raises(NotFittedError, match=err_msg):
        model.transform(data)
    max_features = 2.5
    model.set_params(max_features=max_features)
    with pytest.raises(ValueError, match='`max_features` must be an integer'):
        model.transform(data)