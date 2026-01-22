import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@ignore_warnings
def _check_predict_proba(clf, X, y):
    proba = clf.predict_proba(X)
    log_proba = clf.predict_log_proba(X)
    y = np.atleast_1d(y)
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))
    n_outputs = y.shape[1]
    n_samples = len(X)
    if n_outputs == 1:
        proba = [proba]
        log_proba = [log_proba]
    for k in range(n_outputs):
        assert proba[k].shape[0] == n_samples
        assert proba[k].shape[1] == len(np.unique(y[:, k]))
        assert_array_almost_equal(proba[k].sum(axis=1), np.ones(len(X)))
        assert_array_almost_equal(np.log(proba[k]), log_proba[k])