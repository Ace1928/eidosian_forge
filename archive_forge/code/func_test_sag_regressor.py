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
@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sag_regressor(seed, csr_container):
    """tests if the sag regressor performs well"""
    xmin, xmax = (-5, 5)
    n_samples = 300
    tol = 0.001
    max_iter = 100
    alpha = 0.1
    rng = np.random.RandomState(seed)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)
    y = 0.5 * X.ravel()
    clf1 = Ridge(tol=tol, solver='sag', max_iter=max_iter, alpha=alpha * n_samples, random_state=rng)
    clf2 = clone(clf1)
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    score1 = clf1.score(X, y)
    score2 = clf2.score(X, y)
    assert score1 > 0.98
    assert score2 > 0.98
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()
    clf1 = Ridge(tol=tol, solver='sag', max_iter=max_iter, alpha=alpha * n_samples)
    clf2 = clone(clf1)
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    score1 = clf1.score(X, y)
    score2 = clf2.score(X, y)
    assert score1 > 0.45
    assert score2 > 0.45