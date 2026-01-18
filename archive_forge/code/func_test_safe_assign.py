import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('array_type', ['array', 'sparse', 'dataframe'])
def test_safe_assign(array_type):
    """Check that `_safe_assign` works as expected."""
    rng = np.random.RandomState(0)
    X_array = rng.randn(10, 5)
    row_indexer = [1, 2]
    values = rng.randn(len(row_indexer), X_array.shape[1])
    X = _convert_container(X_array, array_type)
    _safe_assign(X, values, row_indexer=row_indexer)
    assigned_portion = _safe_indexing(X, row_indexer, axis=0)
    assert_allclose_dense_sparse(assigned_portion, _convert_container(values, array_type))
    column_indexer = [1, 2]
    values = rng.randn(X_array.shape[0], len(column_indexer))
    X = _convert_container(X_array, array_type)
    _safe_assign(X, values, column_indexer=column_indexer)
    assigned_portion = _safe_indexing(X, column_indexer, axis=1)
    assert_allclose_dense_sparse(assigned_portion, _convert_container(values, array_type))
    row_indexer, column_indexer = (None, None)
    values = rng.randn(*X.shape)
    X = _convert_container(X_array, array_type)
    _safe_assign(X, values, column_indexer=column_indexer)
    assert_allclose_dense_sparse(X, _convert_container(values, array_type))