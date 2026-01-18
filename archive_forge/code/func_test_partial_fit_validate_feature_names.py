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
@pytest.mark.parametrize('as_frame', [True, False])
def test_partial_fit_validate_feature_names(as_frame):
    """Test that partial_fit from SelectFromModel validates `feature_names_in_`."""
    pytest.importorskip('pandas')
    X, y = datasets.load_iris(as_frame=as_frame, return_X_y=True)
    selector = SelectFromModel(estimator=SGDClassifier(), max_features=4).partial_fit(X, y, classes=[0, 1, 2])
    if as_frame:
        assert_array_equal(selector.feature_names_in_, X.columns)
    else:
        assert not hasattr(selector, 'feature_names_in_')