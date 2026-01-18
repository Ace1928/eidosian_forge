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
@pytest.mark.parametrize('clf', [LogisticRegression(C=len(iris.data), solver='liblinear', multi_class='ovr'), LogisticRegression(C=len(iris.data), solver='lbfgs', multi_class='multinomial'), LogisticRegression(C=len(iris.data), solver='newton-cg', multi_class='multinomial'), LogisticRegression(C=len(iris.data), solver='sag', tol=0.01, multi_class='ovr', random_state=42), LogisticRegression(C=len(iris.data), solver='saga', tol=0.01, multi_class='ovr', random_state=42), LogisticRegression(C=len(iris.data), solver='newton-cholesky', multi_class='ovr')])
def test_predict_iris(clf):
    """Test logistic regression with the iris dataset.

    Test that both multinomial and OvR solvers handle multiclass data correctly and
    give good accuracy score (>0.95) for the training data.
    """
    n_samples, n_features = iris.data.shape
    target = iris.target_names[iris.target]
    if clf.solver == 'lbfgs':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            clf.fit(iris.data, target)
    else:
        clf.fit(iris.data, target)
    assert_array_equal(np.unique(target), clf.classes_)
    pred = clf.predict(iris.data)
    assert np.mean(pred == target) > 0.95
    probabilities = clf.predict_proba(iris.data)
    assert_allclose(probabilities.sum(axis=1), np.ones(n_samples))
    pred = iris.target_names[probabilities.argmax(axis=1)]
    assert np.mean(pred == target) > 0.95