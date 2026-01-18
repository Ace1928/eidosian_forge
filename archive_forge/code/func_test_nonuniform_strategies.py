import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('strategy, expected_2bins, expected_3bins, expected_5bins', [('uniform', [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 2, 2], [0, 0, 1, 1, 4, 4]), ('kmeans', [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 3, 4]), ('quantile', [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 4])])
@pytest.mark.filterwarnings('ignore:In version 1.5 onwards, subsample=200_000')
def test_nonuniform_strategies(strategy, expected_2bins, expected_3bins, expected_5bins):
    X = np.array([0, 0.5, 2, 3, 9, 10]).reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=2, strategy=strategy, encode='ordinal')
    Xt = est.fit_transform(X)
    assert_array_equal(expected_2bins, Xt.ravel())
    est = KBinsDiscretizer(n_bins=3, strategy=strategy, encode='ordinal')
    Xt = est.fit_transform(X)
    assert_array_equal(expected_3bins, Xt.ravel())
    est = KBinsDiscretizer(n_bins=5, strategy=strategy, encode='ordinal')
    Xt = est.fit_transform(X)
    assert_array_equal(expected_5bins, Xt.ravel())