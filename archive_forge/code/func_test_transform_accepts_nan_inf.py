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
def test_transform_accepts_nan_inf():
    clf = NaNTagRandomForest(n_estimators=100, random_state=0)
    nan_data = data.copy()
    model = SelectFromModel(estimator=clf)
    model.fit(nan_data, y)
    nan_data[0] = np.nan
    nan_data[1] = np.inf
    model.transform(nan_data)