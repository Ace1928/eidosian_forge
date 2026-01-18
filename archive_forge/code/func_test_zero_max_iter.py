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
@pytest.mark.parametrize('solver', SOLVERS)
def test_zero_max_iter(solver):
    X, y = load_iris(return_X_y=True)
    y = y == 2
    with ignore_warnings(category=ConvergenceWarning):
        clf = LogisticRegression(solver=solver, max_iter=0).fit(X, y)
    if solver not in ['saga', 'sag']:
        assert clf.n_iter_ == 0
    if solver != 'lbfgs':
        assert_allclose(clf.coef_, np.zeros_like(clf.coef_))
        assert_allclose(clf.decision_function(X), np.full(shape=X.shape[0], fill_value=clf.intercept_))
        assert_allclose(clf.predict_proba(X), np.full(shape=(X.shape[0], 2), fill_value=0.5))
    assert clf.score(X, y) < 0.7