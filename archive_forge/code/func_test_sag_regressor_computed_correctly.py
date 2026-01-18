import math
import re
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import make_dataset
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.linear_model._sag import get_auto_step_size
from sklearn.linear_model._sag_fast import _multinomial_grad_loss_all_samples
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.filterwarnings('ignore:The max_iter was reached')
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sag_regressor_computed_correctly(csr_container):
    """tests if the sag regressor is computed correctly"""
    alpha = 0.1
    n_features = 10
    n_samples = 40
    max_iter = 100
    tol = 1e-06
    fit_intercept = True
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = np.dot(X, w) + 2.0
    step_size = get_step_size(X, alpha, fit_intercept, classification=False)
    clf1 = Ridge(fit_intercept=fit_intercept, tol=tol, solver='sag', alpha=alpha * n_samples, max_iter=max_iter, random_state=rng)
    clf2 = clone(clf1)
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    spweights1, spintercept1 = sag_sparse(X, y, step_size, alpha, n_iter=max_iter, dloss=squared_dloss, fit_intercept=fit_intercept, random_state=rng)
    spweights2, spintercept2 = sag_sparse(X, y, step_size, alpha, n_iter=max_iter, dloss=squared_dloss, sparse=True, fit_intercept=fit_intercept, random_state=rng)
    assert_array_almost_equal(clf1.coef_.ravel(), spweights1.ravel(), decimal=3)
    assert_almost_equal(clf1.intercept_, spintercept1, decimal=1)