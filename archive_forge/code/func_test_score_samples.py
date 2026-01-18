import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_score_samples(global_dtype):
    X_train = np.asarray([[1, 1], [1, 2], [2, 1]], dtype=global_dtype)
    X_test = np.asarray([[2.0, 2.0]], dtype=global_dtype)
    clf1 = neighbors.LocalOutlierFactor(n_neighbors=2, contamination=0.1, novelty=True).fit(X_train)
    clf2 = neighbors.LocalOutlierFactor(n_neighbors=2, novelty=True).fit(X_train)
    clf1_scores = clf1.score_samples(X_test)
    clf1_decisions = clf1.decision_function(X_test)
    clf2_scores = clf2.score_samples(X_test)
    clf2_decisions = clf2.decision_function(X_test)
    assert_allclose(clf1_scores, clf1_decisions + clf1.offset_)
    assert_allclose(clf2_scores, clf2_decisions + clf2.offset_)
    assert_allclose(clf1_scores, clf2_scores)