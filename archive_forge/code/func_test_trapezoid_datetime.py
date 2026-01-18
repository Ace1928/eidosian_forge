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
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
@pytest.mark.parametrize('dask', [True, False])
@pytest.mark.parametrize('which_datetime', ['np', 'cftime'])
def test_trapezoid_datetime(dask, which_datetime) -> None:
    rs = np.random.RandomState(42)
    if which_datetime == 'np':
        coord = np.array(['2004-07-13', '2006-01-13', '2010-08-13', '2010-09-13', '2010-10-11', '2010-12-13', '2011-02-13', '2012-08-13'], dtype='datetime64')
    else:
        if not has_cftime:
            pytest.skip('Test requires cftime.')
        coord = xr.cftime_range('2000', periods=8, freq='2D')
    da = xr.DataArray(rs.randn(8, 6), coords={'time': coord, 'z': 3, 't2d': (('time', 'y'), rs.randn(8, 6))}, dims=['time', 'y'])
    if dask and has_dask:
        da = da.chunk({'time': 4})
    actual = da.integrate('time', datetime_unit='D')
    expected_data = trapezoid(da.compute().data, duck_array_ops.datetime_to_numeric(da['time'].data, datetime_unit='D'), axis=0)
    expected = xr.DataArray(expected_data, dims=['y'], coords={k: v for k, v in da.coords.items() if 'time' not in v.dims})
    assert_allclose(expected, actual.compute())
    assert isinstance(actual.data, type(da.data))
    actual2 = da.integrate('time', datetime_unit='h')
    assert_allclose(actual, actual2 / 24.0)