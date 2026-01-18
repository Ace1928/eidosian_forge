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
def test_safe_indexing_container_axis_0_unsupported_type():
    indices = ['col_1', 'col_2']
    array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    err_msg = "String indexing is not supported with 'axis=0'"
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(array, indices, axis=0)