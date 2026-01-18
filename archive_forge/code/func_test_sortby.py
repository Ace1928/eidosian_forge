from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_sortby(self) -> None:
    ds = Dataset({'A': DataArray([[1, 2], [3, 4], [5, 6]], [('x', ['c', 'b', 'a']), ('y', [1, 0])]), 'B': DataArray([[5, 6], [7, 8], [9, 10]], dims=['x', 'y'])})
    sorted1d = Dataset({'A': DataArray([[5, 6], [3, 4], [1, 2]], [('x', ['a', 'b', 'c']), ('y', [1, 0])]), 'B': DataArray([[9, 10], [7, 8], [5, 6]], dims=['x', 'y'])})
    sorted2d = Dataset({'A': DataArray([[6, 5], [4, 3], [2, 1]], [('x', ['a', 'b', 'c']), ('y', [0, 1])]), 'B': DataArray([[10, 9], [8, 7], [6, 5]], dims=['x', 'y'])})
    expected = sorted1d
    dax = DataArray([100, 99, 98], [('x', ['c', 'b', 'a'])])
    actual = ds.sortby(dax)
    assert_equal(actual, expected)
    actual = ds.sortby(dax, ascending=False)
    assert_equal(actual, ds)
    dax_short = DataArray([98, 97], [('x', ['b', 'a'])])
    actual = ds.sortby(dax_short)
    assert_equal(actual, expected)
    dax0 = DataArray([100, 95, 95], [('x', ['c', 'b', 'a'])])
    dax1 = DataArray([0, 1, 0], [('x', ['c', 'b', 'a'])])
    actual = ds.sortby([dax0, dax1])
    assert_equal(actual, expected)
    expected = sorted2d
    day = DataArray([90, 80], [('y', [1, 0])])
    actual = ds.sortby([day, dax])
    assert_equal(actual, expected)
    with pytest.raises(KeyError):
        actual = ds.sortby('z')
    with pytest.raises(ValueError) as excinfo:
        actual = ds.sortby(ds['A'])
    assert 'DataArray is not 1-D' in str(excinfo.value)
    expected = sorted1d
    actual = ds.sortby('x')
    assert_equal(actual, expected)
    indices = (('b', 1), ('b', 0), ('a', 1), ('a', 0))
    midx = pd.MultiIndex.from_tuples(indices, names=['one', 'two'])
    ds_midx = Dataset({'A': DataArray([[1, 2], [3, 4], [5, 6], [7, 8]], [('x', midx), ('y', [1, 0])]), 'B': DataArray([[5, 6], [7, 8], [9, 10], [11, 12]], dims=['x', 'y'])})
    actual = ds_midx.sortby('x')
    midx_reversed = pd.MultiIndex.from_tuples(tuple(reversed(indices)), names=['one', 'two'])
    expected = Dataset({'A': DataArray([[7, 8], [5, 6], [3, 4], [1, 2]], [('x', midx_reversed), ('y', [1, 0])]), 'B': DataArray([[11, 12], [9, 10], [7, 8], [5, 6]], dims=['x', 'y'])})
    assert_equal(actual, expected)
    expected = sorted2d
    actual = ds.sortby(['x', 'y'])
    assert_equal(actual, expected)
    actual = ds.sortby(['x', 'y'], ascending=False)
    assert_equal(actual, ds)