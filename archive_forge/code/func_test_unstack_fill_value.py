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
def test_unstack_fill_value(self) -> None:
    ds = xr.Dataset({'var': (('x',), np.arange(6)), 'other_var': (('x',), np.arange(3, 9))}, coords={'x': [0, 1, 2] * 2, 'y': (('x',), ['a'] * 3 + ['b'] * 3)})
    ds = ds.isel(x=[0, 2, 3, 4]).set_index(index=['x', 'y'])
    actual1 = ds.unstack('index', fill_value=-1)
    expected1 = ds.unstack('index').fillna(-1).astype(int)
    assert actual1['var'].dtype == int
    assert_equal(actual1, expected1)
    actual2 = ds['var'].unstack('index', fill_value=-1)
    expected2 = ds['var'].unstack('index').fillna(-1).astype(int)
    assert_equal(actual2, expected2)
    actual3 = ds.unstack('index', fill_value={'var': -1, 'other_var': 1})
    expected3 = ds.unstack('index').fillna({'var': -1, 'other_var': 1}).astype(int)
    assert_equal(actual3, expected3)