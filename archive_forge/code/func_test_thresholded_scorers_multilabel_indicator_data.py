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
def test_thresholded_scorers_multilabel_indicator_data():
    X, y = make_multilabel_classification(allow_unlabeled=False, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)
    score1 = get_scorer('roc_auc')(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, np.vstack([p[:, -1] for p in y_proba]).T)
    assert_almost_equal(score1, score2)
    clf = OneVsRestClassifier(DecisionTreeClassifier())
    clf.fit(X_train, y_train)
    score1 = get_scorer('roc_auc')(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.predict_proba(X_test))
    assert_almost_equal(score1, score2)
    clf = OneVsRestClassifier(LinearSVC(dual='auto', random_state=0))
    clf.fit(X_train, y_train)
    score1 = get_scorer('roc_auc')(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.decision_function(X_test))
    assert_almost_equal(score1, score2)