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
@pytest.mark.parametrize('X_constructor', ['array', 'series'])
def test_safe_indexing_1d_array_error(X_constructor):
    X = list(range(5))
    if X_constructor == 'array':
        X_constructor = np.asarray(X)
    elif X_constructor == 'series':
        pd = pytest.importorskip('pandas')
        X_constructor = pd.Series(X)
    err_msg = "'X' should be a 2D NumPy array, 2D sparse matrix or pandas"
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(X_constructor, [0, 1], axis=1)