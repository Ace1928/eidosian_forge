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
def test_where_drop(self) -> None:
    array = DataArray(range(5), coords=[range(5)], dims=['x'])
    expected1 = DataArray(range(5)[2:], coords=[range(5)[2:]], dims=['x'])
    actual1 = array.where(array > 1, drop=True)
    assert_identical(expected1, actual1)
    ds = Dataset({'a': array})
    expected2 = Dataset({'a': expected1})
    actual2 = ds.where(ds > 1, drop=True)
    assert_identical(expected2, actual2)
    actual3 = ds.where(ds.a > 1, drop=True)
    assert_identical(expected2, actual3)
    with pytest.raises(TypeError, match='must be a'):
        ds.where(np.arange(5) > 1, drop=True)
    array = DataArray(np.array([2, 7, 1, 8, 3]), coords=[np.array([3, 1, 4, 5, 9])], dims=['x'])
    expected4 = DataArray(np.array([7, 8, 3]), coords=[np.array([1, 5, 9])], dims=['x'])
    actual4 = array.where(array > 2, drop=True)
    assert_identical(expected4, actual4)
    ds = Dataset({'a': ('x', [0, 1, 2, 3]), 'b': ('x', [4, 5, 6, 7])})
    expected5 = Dataset({'a': ('x', [np.nan, 1, 2, 3]), 'b': ('x', [4, 5, 6, np.nan])})
    actual5 = ds.where((ds > 0) & (ds < 7), drop=True)
    assert_identical(expected5, actual5)
    ds = Dataset({'a': (('x', 'y'), [[0, 1], [2, 3]])})
    expected6 = Dataset({'a': (('x', 'y'), [[np.nan, 1], [2, 3]])})
    actual6 = ds.where(ds > 0, drop=True)
    assert_identical(expected6, actual6)
    ds = Dataset({'a': (('x', 'y'), [[0, 1], [2, 3]])}, coords={'x': [4, 3], 'y': [1, 2], 'z': (['x', 'y'], [[np.e, np.pi], [np.pi * np.e, np.pi * 3]])})
    expected7 = Dataset({'a': (('x', 'y'), [[3]])}, coords={'x': [3], 'y': [2], 'z': (['x', 'y'], [[np.pi * 3]])})
    actual7 = ds.where(ds > 2, drop=True)
    assert_identical(expected7, actual7)
    ds = Dataset({'a': (('x', 'y'), [[0, 1], [2, 3]]), 'b': (('x', 'y'), [[4, 5], [6, 7]])})
    expected8 = Dataset({'a': (('x', 'y'), [[np.nan, 1], [2, 3]]), 'b': (('x', 'y'), [[4, 5], [6, 7]])})
    actual8 = ds.where(ds > 0, drop=True)
    assert_identical(expected8, actual8)
    ds = xr.Dataset({'a': ('x', [1, 2, 3]), 'b': ('y', [2, 3, 4]), 'c': (('x', 'y'), np.arange(9).reshape((3, 3)))})
    expected9 = xr.Dataset({'a': ('x', [np.nan, 3]), 'b': ('y', [np.nan, 3, 4]), 'c': (('x', 'y'), np.arange(3.0, 9.0).reshape((2, 3)))})
    actual9 = ds.where(ds > 2, drop=True)
    assert actual9.sizes['x'] == 2
    assert_identical(expected9, actual9)