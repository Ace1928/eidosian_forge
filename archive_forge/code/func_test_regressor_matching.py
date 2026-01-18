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
def test_regressor_matching():
    n_samples = 10
    n_features = 5
    rng = np.random.RandomState(10)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    y = X.dot(true_w)
    alpha = 1.0
    n_iter = 100
    fit_intercept = True
    step_size = get_step_size(X, alpha, fit_intercept, classification=False)
    clf = Ridge(fit_intercept=fit_intercept, tol=1e-11, solver='sag', alpha=alpha * n_samples, max_iter=n_iter)
    clf.fit(X, y)
    weights1, intercept1 = sag_sparse(X, y, step_size, alpha, n_iter=n_iter, dloss=squared_dloss, fit_intercept=fit_intercept)
    weights2, intercept2 = sag(X, y, step_size, alpha, n_iter=n_iter, dloss=squared_dloss, fit_intercept=fit_intercept)
    assert_allclose(weights1, clf.coef_)
    assert_allclose(intercept1, clf.intercept_)
    assert_allclose(weights2, clf.coef_)
    assert_allclose(intercept2, clf.intercept_)