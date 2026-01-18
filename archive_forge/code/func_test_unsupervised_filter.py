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
@pytest.mark.parametrize('selector', [SelectKBest(k=4), SelectPercentile(percentile=80), GenericUnivariateSelect(mode='k_best', param=4), GenericUnivariateSelect(mode='percentile', param=80)])
def test_unsupervised_filter(selector):
    """Check support for unsupervised feature selection for the filter that could
    require only `X`.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)

    def score_func(X, y=None):
        return np.array([1, 1, 1, 1, 0])
    selector.set_params(score_func=score_func)
    selector.fit(X)
    X_trans = selector.transform(X)
    assert_allclose(X_trans, X[:, :4])
    X_trans = selector.fit_transform(X)
    assert_allclose(X_trans, X[:, :4])