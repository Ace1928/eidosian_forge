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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_saga_vs_liblinear(csr_container):
    iris = load_iris()
    X, y = (iris.data, iris.target)
    X = np.concatenate([X] * 3)
    y = np.concatenate([y] * 3)
    X_bin = X[y <= 1]
    y_bin = y[y <= 1] * 2 - 1
    X_sparse, y_sparse = make_classification(n_samples=50, n_features=20, random_state=0)
    X_sparse = csr_container(X_sparse)
    for X, y in ((X_bin, y_bin), (X_sparse, y_sparse)):
        for penalty in ['l1', 'l2']:
            n_samples = X.shape[0]
            for alpha in np.logspace(-1, 1, 3):
                saga = LogisticRegression(C=1.0 / (n_samples * alpha), solver='saga', multi_class='ovr', max_iter=200, fit_intercept=False, penalty=penalty, random_state=0, tol=1e-06)
                liblinear = LogisticRegression(C=1.0 / (n_samples * alpha), solver='liblinear', multi_class='ovr', max_iter=200, fit_intercept=False, penalty=penalty, random_state=0, tol=1e-06)
                saga.fit(X, y)
                liblinear.fit(X, y)
                assert_array_almost_equal(saga.coef_, liblinear.coef_, 3)