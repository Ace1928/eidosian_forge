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
def test_virtual_variables_time(self) -> None:
    data = create_test_data()
    assert_array_equal(data['time.month'].values, data.variables['time'].to_index().month)
    assert_array_equal(data['time.season'].values, 'DJF')
    assert_array_equal(data['time.dayofyear'] + 1, 2 + np.arange(20))
    assert_array_equal(np.sin(data['time.dayofyear']), np.sin(1 + np.arange(20)))
    expected = Dataset({}, {'dayofyear': data['time.dayofyear']})
    actual = data[['time.dayofyear']]
    assert_equal(expected, actual)
    ds = Dataset({'t': ('x', pd.date_range('2000-01-01', periods=3))})
    assert (ds['t.year'] == 2000).all()