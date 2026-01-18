import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_select_percentile_regression():
    X, y = make_regression(n_samples=200, n_features=20, n_informative=5, shuffle=False, random_state=0)
    univariate_filter = SelectPercentile(f_regression, percentile=25)
    X_r = univariate_filter.fit(X, y).transform(X)
    assert_best_scores_kept(univariate_filter)
    X_r2 = GenericUnivariateSelect(f_regression, mode='percentile', param=25).fit(X, y).transform(X)
    assert_array_equal(X_r, X_r2)
    support = univariate_filter.get_support()
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    assert_array_equal(support, gtruth)
    X_2 = X.copy()
    X_2[:, np.logical_not(support)] = 0
    assert_array_equal(X_2, univariate_filter.inverse_transform(X_r))
    assert_array_equal(X_2.astype(bool), univariate_filter.inverse_transform(X_r.astype(bool)))