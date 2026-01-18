import functools
import warnings
from typing import Any, List
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.random_projection import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
def test_sparse_random_matrix():
    n_components = 100
    n_features = 500
    for density in [0.3, 1.0]:
        s = 1 / density
        A = _sparse_random_matrix(n_components, n_features, density=density, random_state=0)
        A = densify(A)
        values = np.unique(A)
        assert np.sqrt(s) / np.sqrt(n_components) in values
        assert -np.sqrt(s) / np.sqrt(n_components) in values
        if density == 1.0:
            assert np.size(values) == 2
        else:
            assert 0.0 in values
            assert np.size(values) == 3
        assert_almost_equal(np.mean(A == 0.0), 1 - 1 / s, decimal=2)
        assert_almost_equal(np.mean(A == np.sqrt(s) / np.sqrt(n_components)), 1 / (2 * s), decimal=2)
        assert_almost_equal(np.mean(A == -np.sqrt(s) / np.sqrt(n_components)), 1 / (2 * s), decimal=2)
        assert_almost_equal(np.var(A == 0.0, ddof=1), (1 - 1 / s) * 1 / s, decimal=2)
        assert_almost_equal(np.var(A == np.sqrt(s) / np.sqrt(n_components), ddof=1), (1 - 1 / (2 * s)) * 1 / (2 * s), decimal=2)
        assert_almost_equal(np.var(A == -np.sqrt(s) / np.sqrt(n_components), ddof=1), (1 - 1 / (2 * s)) * 1 / (2 * s), decimal=2)