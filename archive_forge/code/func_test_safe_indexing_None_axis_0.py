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
@pytest.mark.parametrize('array_type', ['list', 'array', 'sparse'])
def test_safe_indexing_None_axis_0(array_type):
    X = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type)
    X_subset = _safe_indexing(X, None, axis=0)
    assert_allclose_dense_sparse(X_subset, X)