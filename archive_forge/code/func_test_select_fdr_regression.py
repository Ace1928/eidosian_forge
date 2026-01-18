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
@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1])
@pytest.mark.parametrize('n_informative', [1, 5, 10])
def test_select_fdr_regression(alpha, n_informative):

    def single_fdr(alpha, n_informative, random_state):
        X, y = make_regression(n_samples=150, n_features=20, n_informative=n_informative, shuffle=False, random_state=random_state, noise=10)
        with warnings.catch_warnings(record=True):
            univariate_filter = SelectFdr(f_regression, alpha=alpha)
            X_r = univariate_filter.fit(X, y).transform(X)
            X_r2 = GenericUnivariateSelect(f_regression, mode='fdr', param=alpha).fit(X, y).transform(X)
        assert_array_equal(X_r, X_r2)
        support = univariate_filter.get_support()
        num_false_positives = np.sum(support[n_informative:] == 1)
        num_true_positives = np.sum(support[:n_informative] == 1)
        if num_false_positives == 0:
            return 0.0
        false_discovery_rate = num_false_positives / (num_true_positives + num_false_positives)
        return false_discovery_rate
    false_discovery_rate = np.mean([single_fdr(alpha, n_informative, random_state) for random_state in range(100)])
    assert alpha >= false_discovery_rate
    if false_discovery_rate != 0:
        assert false_discovery_rate > alpha / 10