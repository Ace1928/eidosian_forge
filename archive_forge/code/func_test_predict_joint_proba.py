import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('Estimator', ALL_NAIVE_BAYES_CLASSES)
def test_predict_joint_proba(Estimator, global_random_seed):
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)
    est = Estimator().fit(X2, y2)
    jll = est.predict_joint_log_proba(X2)
    log_prob_x = logsumexp(jll, axis=1)
    log_prob_x_y = jll - np.atleast_2d(log_prob_x).T
    assert_allclose(est.predict_log_proba(X2), log_prob_x_y)