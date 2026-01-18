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
def test_step_size_alpha_error():
    X = [[0, 0], [0, 0]]
    y = [1, -1]
    fit_intercept = False
    alpha = 1.0
    msg = re.escape('Current sag implementation does not handle the case step_size * alpha_scaled == 1')
    clf1 = LogisticRegression(solver='sag', C=1.0 / alpha, fit_intercept=fit_intercept)
    with pytest.raises(ZeroDivisionError, match=msg):
        clf1.fit(X, y)
    clf2 = Ridge(fit_intercept=fit_intercept, solver='sag', alpha=alpha)
    with pytest.raises(ZeroDivisionError, match=msg):
        clf2.fit(X, y)