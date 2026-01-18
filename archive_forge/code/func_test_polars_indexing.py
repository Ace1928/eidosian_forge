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
def test_polars_indexing():
    """Check _safe_indexing for polars as expected."""
    pl = pytest.importorskip('polars', minversion='0.18.2')
    df = pl.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 5, 6, 8], 'c': [1, 4, 1, 10]}, orient='row')
    from polars.testing import assert_frame_equal
    str_keys = [['b'], ['a', 'b'], ['b', 'a', 'c'], ['c'], ['a']]
    for key in str_keys:
        out = _safe_indexing(df, key, axis=1)
        assert_frame_equal(df[key], out)
    bool_keys = [([True, False, True], ['a', 'c']), ([False, False, True], ['c'])]
    for bool_key, str_key in bool_keys:
        out = _safe_indexing(df, bool_key, axis=1)
        assert_frame_equal(df[:, str_key], out)
    int_keys = [([0, 1], ['a', 'b']), ([2], ['c'])]
    for int_key, str_key in int_keys:
        out = _safe_indexing(df, int_key, axis=1)
        assert_frame_equal(df[:, str_key], out)
    axis_0_keys = [[0, 1], [1, 3], [3, 2]]
    for key in axis_0_keys:
        out = _safe_indexing(df, key, axis=0)
        assert_frame_equal(df[key], out)