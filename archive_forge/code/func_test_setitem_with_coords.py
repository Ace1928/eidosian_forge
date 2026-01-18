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
def test_setitem_with_coords(self) -> None:
    ds = create_test_data()
    other = DataArray(np.arange(10), dims='dim3', coords={'numbers': ('dim3', np.arange(10))})
    expected = ds.copy()
    expected['var3'] = other.drop_vars('numbers')
    actual = ds.copy()
    actual['var3'] = other
    assert_identical(expected, actual)
    assert 'numbers' in other.coords
    other = ds['var3'].isel(dim3=slice(1, -1))
    other['numbers'] = ('dim3', np.arange(8))
    actual = ds.copy()
    actual['var3'] = other
    assert 'numbers' in other.coords
    expected = ds.copy()
    expected['var3'] = ds['var3'].isel(dim3=slice(1, -1))
    assert_identical(expected, actual)
    other = ds['var3'].isel(dim3=slice(1, -1))
    other['numbers'] = ('dim3', np.arange(8))
    other['position'] = ('dim3', np.arange(8))
    actual = ds.copy()
    actual['var3'] = other
    assert 'position' in actual
    assert 'position' in other.coords
    actual = ds.copy()
    other = actual['numbers']
    other[0] = 10
    actual['numbers'] = other
    assert actual['numbers'][0] == 10
    ds = Dataset({'var': ('x', [1, 2, 3])}, coords={'x': [0, 1, 2], 'z1': ('x', [1, 2, 3]), 'z2': ('x', [1, 2, 3])})
    ds['var'] = ds['var'] * 2
    assert np.allclose(ds['var'], [2, 4, 6])