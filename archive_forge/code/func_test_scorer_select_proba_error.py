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
@pytest.mark.parametrize('scorer', [make_scorer(average_precision_score, response_method=('decision_function', 'predict_proba'), pos_label='xxx'), make_scorer(brier_score_loss, response_method='predict_proba', pos_label='xxx'), make_scorer(f1_score, pos_label='xxx')], ids=['non-thresholded scorer', 'probability scorer', 'thresholded scorer'])
def test_scorer_select_proba_error(scorer):
    X, y = make_classification(n_classes=2, n_informative=3, n_samples=20, random_state=0)
    lr = LogisticRegression().fit(X, y)
    assert scorer._kwargs['pos_label'] not in np.unique(y).tolist()
    err_msg = 'is not a valid label'
    with pytest.raises(ValueError, match=err_msg):
        scorer(lr, X, y)