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
def test_set_index_deindexed_coords(self) -> None:
    one = ['a', 'a', 'b', 'b']
    two = [1, 2, 1, 2]
    three = ['c', 'c', 'd', 'd']
    four = [3, 4, 3, 4]
    midx_12 = pd.MultiIndex.from_arrays([one, two], names=['one', 'two'])
    midx_34 = pd.MultiIndex.from_arrays([three, four], names=['three', 'four'])
    coords = Coordinates.from_pandas_multiindex(midx_12, 'x')
    coords['three'] = ('x', three)
    coords['four'] = ('x', four)
    ds = xr.Dataset(coords=coords)
    actual = ds.set_index(x=['three', 'four'])
    coords_expected = Coordinates.from_pandas_multiindex(midx_34, 'x')
    coords_expected['one'] = ('x', one)
    coords_expected['two'] = ('x', two)
    expected = xr.Dataset(coords=coords_expected)
    assert_identical(actual, expected)