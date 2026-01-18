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
@ignore_warnings
def test_raises_on_score_list():
    X, y = make_blobs(random_state=0)
    f1_scorer_no_average = make_scorer(f1_score, average=None)
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        cross_val_score(clf, X, y, scoring=f1_scorer_no_average)
    grid_search = GridSearchCV(clf, scoring=f1_scorer_no_average, param_grid={'max_depth': [1, 2]})
    with pytest.raises(ValueError):
        grid_search.fit(X, y)