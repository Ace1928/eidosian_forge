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
def test_sag_pobj_matches_logistic_regression(csr_container):
    """tests if the sag pobj matches log reg"""
    n_samples = 100
    alpha = 1.0
    max_iter = 20
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.1)
    clf1 = LogisticRegression(solver='sag', fit_intercept=False, tol=1e-07, C=1.0 / alpha / n_samples, max_iter=max_iter, random_state=10, multi_class='ovr')
    clf2 = clone(clf1)
    clf3 = LogisticRegression(fit_intercept=False, tol=1e-07, C=1.0 / alpha / n_samples, max_iter=max_iter, random_state=10, multi_class='ovr')
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    clf3.fit(X, y)
    pobj1 = get_pobj(clf1.coef_, alpha, X, y, log_loss)
    pobj2 = get_pobj(clf2.coef_, alpha, X, y, log_loss)
    pobj3 = get_pobj(clf3.coef_, alpha, X, y, log_loss)
    assert_array_almost_equal(pobj1, pobj2, decimal=4)
    assert_array_almost_equal(pobj2, pobj3, decimal=4)
    assert_array_almost_equal(pobj3, pobj1, decimal=4)