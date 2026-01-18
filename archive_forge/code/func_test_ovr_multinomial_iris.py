import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_ovr_multinomial_iris():
    train, target = (iris.data, iris.target)
    n_samples, n_features = train.shape
    n_cv = 2
    cv = StratifiedKFold(n_cv)
    precomputed_folds = list(cv.split(train, target))
    clf = LogisticRegressionCV(cv=precomputed_folds, multi_class='ovr')
    clf.fit(train, target)
    clf1 = LogisticRegressionCV(cv=precomputed_folds, multi_class='ovr')
    target_copy = target.copy()
    target_copy[target_copy == 0] = 1
    clf1.fit(train, target_copy)
    assert_allclose(clf.scores_[2], clf1.scores_[2])
    assert_allclose(clf.intercept_[2:], clf1.intercept_)
    assert_allclose(clf.coef_[2][np.newaxis, :], clf1.coef_)
    assert clf.coef_.shape == (3, n_features)
    assert_array_equal(clf.classes_, [0, 1, 2])
    coefs_paths = np.asarray(list(clf.coefs_paths_.values()))
    assert coefs_paths.shape == (3, n_cv, 10, n_features + 1)
    assert clf.Cs_.shape == (10,)
    scores = np.asarray(list(clf.scores_.values()))
    assert scores.shape == (3, n_cv, 10)
    for solver in ['lbfgs', 'newton-cg', 'sag', 'saga']:
        max_iter = 500 if solver in ['sag', 'saga'] else 30
        clf_multi = LogisticRegressionCV(solver=solver, multi_class='multinomial', max_iter=max_iter, random_state=42, tol=0.001 if solver in ['sag', 'saga'] else 0.01, cv=2)
        if solver == 'lbfgs':
            train = scale(train)
        clf_multi.fit(train, target)
        multi_score = clf_multi.score(train, target)
        ovr_score = clf.score(train, target)
        assert multi_score > ovr_score
        assert clf.coef_.shape == clf_multi.coef_.shape
        assert_array_equal(clf_multi.classes_, [0, 1, 2])
        coefs_paths = np.asarray(list(clf_multi.coefs_paths_.values()))
        assert coefs_paths.shape == (3, n_cv, 10, n_features + 1)
        assert clf_multi.Cs_.shape == (10,)
        scores = np.asarray(list(clf_multi.scores_.values()))
        assert scores.shape == (3, n_cv, 10)