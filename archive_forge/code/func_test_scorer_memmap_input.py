import numbers
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge
from sklearn.metrics import (
from sklearn.metrics import cluster as cluster_module
from sklearn.metrics._scorer import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
from sklearn.utils.metadata_routing import MetadataRouter
@pytest.mark.parametrize('name', get_scorer_names())
def test_scorer_memmap_input(name):
    if name in REQUIRE_POSITIVE_Y_SCORERS:
        y_mm_1 = _require_positive_y(y_mm)
        y_ml_mm_1 = _require_positive_y(y_ml_mm)
    else:
        y_mm_1, y_ml_mm_1 = (y_mm, y_ml_mm)
    with ignore_warnings():
        scorer, estimator = (get_scorer(name), ESTIMATORS[name])
        if name in MULTILABEL_ONLY_SCORERS:
            score = scorer(estimator, X_mm, y_ml_mm_1)
        else:
            score = scorer(estimator, X_mm, y_mm_1)
        assert isinstance(score, numbers.Number), name