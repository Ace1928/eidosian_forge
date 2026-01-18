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
@pytest.mark.filterwarnings('ignore:Once the behaviour of DataArray:DeprecationWarning')
def test_reduce_strings(self) -> None:
    expected = Dataset({'x': 'a'})
    ds = Dataset({'x': ('y', ['a', 'b'])})
    ds.coords['y'] = [-10, 10]
    actual = ds.min()
    assert_identical(expected, actual)
    expected = Dataset({'x': 'b'})
    actual = ds.max()
    assert_identical(expected, actual)
    expected = Dataset({'x': 0})
    actual = ds.argmin()
    assert_identical(expected, actual)
    expected = Dataset({'x': 1})
    actual = ds.argmax()
    assert_identical(expected, actual)
    expected = Dataset({'x': -10})
    actual = ds.idxmin()
    assert_identical(expected, actual)
    expected = Dataset({'x': 10})
    actual = ds.idxmax()
    assert_identical(expected, actual)
    expected = Dataset({'x': b'a'})
    ds = Dataset({'x': ('y', np.array(['a', 'b'], 'S1'))})
    actual = ds.min()
    assert_identical(expected, actual)
    expected = Dataset({'x': 'a'})
    ds = Dataset({'x': ('y', np.array(['a', 'b'], 'U1'))})
    actual = ds.min()
    assert_identical(expected, actual)