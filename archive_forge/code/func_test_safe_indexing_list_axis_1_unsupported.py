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
@pytest.mark.parametrize('indices', [0, [0, 1], slice(0, 2), np.array([0, 1])])
def test_safe_indexing_list_axis_1_unsupported(indices):
    """Check that we raise a ValueError when axis=1 with input as list."""
    X = [[1, 2], [4, 5], [7, 8]]
    err_msg = 'axis=1 is not supported for lists'
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(X, indices, axis=1)