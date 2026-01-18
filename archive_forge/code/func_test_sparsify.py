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
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_sparsify(coo_container):
    n_samples, n_features = iris.data.shape
    target = iris.target_names[iris.target]
    X = scale(iris.data)
    clf = LogisticRegression(random_state=0).fit(X, target)
    pred_d_d = clf.decision_function(X)
    clf.sparsify()
    assert sparse.issparse(clf.coef_)
    pred_s_d = clf.decision_function(X)
    sp_data = coo_container(X)
    pred_s_s = clf.decision_function(sp_data)
    clf.densify()
    pred_d_s = clf.decision_function(sp_data)
    assert_array_almost_equal(pred_d_d, pred_s_d)
    assert_array_almost_equal(pred_d_d, pred_s_s)
    assert_array_almost_equal(pred_d_d, pred_d_s)