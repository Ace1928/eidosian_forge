import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('i', range(1, 9))
def test_numeric_stability(i):
    X_init = np.array([2.0, 4.0, 6.0, 8.0, 10.0]).reshape(-1, 1)
    Xt_expected = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)
    X = X_init / 10 ** i
    Xt = KBinsDiscretizer(n_bins=2, encode='ordinal').fit_transform(X)
    assert_array_equal(Xt_expected, Xt)