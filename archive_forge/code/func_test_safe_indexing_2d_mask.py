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
@pytest.mark.parametrize('indices_type', ['list', 'tuple', 'array', 'series'])
@pytest.mark.parametrize('axis, expected_subset', [(0, [[4, 5, 6], [7, 8, 9]]), (1, [[2, 3], [5, 6], [8, 9]])])
def test_safe_indexing_2d_mask(array_type, indices_type, axis, expected_subset):
    columns_name = ['col_0', 'col_1', 'col_2']
    array = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name)
    indices = [False, True, True]
    indices = _convert_container(indices, indices_type)
    subset = _safe_indexing(array, indices, axis=axis)
    assert_allclose_dense_sparse(subset, _convert_container(expected_subset, array_type))