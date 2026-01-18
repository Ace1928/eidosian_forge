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
def test_select_kbest_all():
    X, y = make_classification(n_samples=20, n_features=10, shuffle=False, random_state=0)
    univariate_filter = SelectKBest(f_classif, k='all')
    X_r = univariate_filter.fit(X, y).transform(X)
    assert_array_equal(X, X_r)
    X_r2 = GenericUnivariateSelect(f_classif, mode='k_best', param='all').fit(X, y).transform(X)
    assert_array_equal(X_r, X_r2)