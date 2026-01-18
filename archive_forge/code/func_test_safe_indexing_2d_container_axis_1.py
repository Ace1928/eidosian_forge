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
@pytest.mark.parametrize('indices_type', ['list', 'tuple', 'array', 'series', 'slice'])
@pytest.mark.parametrize('indices', [[1, 2], ['col_1', 'col_2']])
def test_safe_indexing_2d_container_axis_1(array_type, indices_type, indices):
    indices_converted = copy(indices)
    if indices_type == 'slice' and isinstance(indices[1], int):
        indices_converted[1] += 1
    columns_name = ['col_0', 'col_1', 'col_2']
    array = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name)
    indices_converted = _convert_container(indices_converted, indices_type)
    if isinstance(indices[0], str) and array_type != 'dataframe':
        err_msg = 'Specifying the columns using strings is only supported for dataframes'
        with pytest.raises(ValueError, match=err_msg):
            _safe_indexing(array, indices_converted, axis=1)
    else:
        subset = _safe_indexing(array, indices_converted, axis=1)
        assert_allclose_dense_sparse(subset, _convert_container([[2, 3], [5, 6], [8, 9]], array_type))