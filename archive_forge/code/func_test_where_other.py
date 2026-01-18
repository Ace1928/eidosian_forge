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
def test_where_other(self) -> None:
    ds = Dataset({'a': ('x', range(5))}, {'x': range(5)})
    expected = Dataset({'a': ('x', [-1, -1, 2, 3, 4])}, {'x': range(5)})
    actual = ds.where(ds > 1, -1)
    assert_equal(expected, actual)
    assert actual.a.dtype == int
    actual = ds.where(lambda x: x > 1, -1)
    assert_equal(expected, actual)
    actual = ds.where(ds > 1, other=-1, drop=True)
    expected_nodrop = ds.where(ds > 1, -1)
    _, expected = xr.align(actual, expected_nodrop, join='left')
    assert_equal(actual, expected)
    assert actual.a.dtype == int
    with pytest.raises(ValueError, match='cannot align .* are not equal'):
        ds.where(ds > 1, ds.isel(x=slice(3)))
    with pytest.raises(ValueError, match='exact match required'):
        ds.where(ds > 1, ds.assign(b=2))