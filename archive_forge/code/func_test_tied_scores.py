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
def test_tied_scores():
    X_train = np.array([[0, 0, 0], [1, 1, 1]])
    y_train = [0, 1]
    for n_features in [1, 2, 3]:
        sel = SelectKBest(chi2, k=n_features).fit(X_train, y_train)
        X_test = sel.transform([[0, 1, 2]])
        assert_array_equal(X_test[0], np.arange(3)[-n_features:])