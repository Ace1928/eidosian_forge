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
@pytest.mark.parametrize('dtype_in', [np.float32, np.float64])
def test_select_kbest_zero(dtype_in):
    X, y = make_classification(n_samples=20, n_features=10, shuffle=False, random_state=0)
    X = X.astype(dtype_in)
    univariate_filter = SelectKBest(f_classif, k=0)
    univariate_filter.fit(X, y)
    support = univariate_filter.get_support()
    gtruth = np.zeros(10, dtype=bool)
    assert_array_equal(support, gtruth)
    with pytest.warns(UserWarning, match='No features were selected'):
        X_selected = univariate_filter.transform(X)
    assert X_selected.shape == (20, 0)
    assert X_selected.dtype == dtype_in