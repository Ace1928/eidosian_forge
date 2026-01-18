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
def test_swap_dims(self) -> None:
    original = Dataset({'x': [1, 2, 3], 'y': ('x', list('abc')), 'z': 42})
    expected = Dataset({'z': 42}, {'x': ('y', [1, 2, 3]), 'y': list('abc')})
    actual = original.swap_dims({'x': 'y'})
    assert_identical(expected, actual)
    assert isinstance(actual.variables['y'], IndexVariable)
    assert isinstance(actual.variables['x'], Variable)
    assert actual.xindexes['y'].equals(expected.xindexes['y'])
    roundtripped = actual.swap_dims({'y': 'x'})
    assert_identical(original.set_coords('y'), roundtripped)
    with pytest.raises(ValueError, match='cannot swap'):
        original.swap_dims({'y': 'x'})
    with pytest.raises(ValueError, match='replacement dimension'):
        original.swap_dims({'x': 'z'})
    expected = Dataset({'y': ('u', list('abc')), 'z': 42}, coords={'x': ('u', [1, 2, 3])})
    actual = original.swap_dims({'x': 'u'})
    assert_identical(expected, actual)
    expected = Dataset({'y': ('u', list('abc')), 'z': 42}, coords={'x': ('u', [1, 2, 3])})
    actual = original.swap_dims(x='u')
    assert_identical(expected, actual)
    midx = pd.MultiIndex.from_arrays([list('aab'), list('yzz')], names=['y1', 'y2'])
    original = Dataset({'x': [1, 2, 3], 'y': ('x', midx), 'z': 42})
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'y')
    midx_coords['x'] = ('y', [1, 2, 3])
    expected = Dataset({'z': 42}, midx_coords)
    actual = original.swap_dims({'x': 'y'})
    assert_identical(expected, actual)
    assert isinstance(actual.variables['y'], IndexVariable)
    assert isinstance(actual.variables['x'], Variable)
    assert actual.xindexes['y'].equals(expected.xindexes['y'])