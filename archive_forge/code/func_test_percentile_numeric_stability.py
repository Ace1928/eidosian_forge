import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
def test_percentile_numeric_stability():
    X = np.array([0.05, 0.05, 0.95]).reshape(-1, 1)
    bin_edges = np.array([0.05, 0.23, 0.41, 0.59, 0.77, 0.95])
    Xt = np.array([0, 0, 4]).reshape(-1, 1)
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    warning_message = 'Consider decreasing the number of bins.'
    with pytest.warns(UserWarning, match=warning_message):
        kbd.fit(X)
    assert_array_almost_equal(kbd.bin_edges_[0], bin_edges)
    assert_array_almost_equal(kbd.transform(X), Xt)