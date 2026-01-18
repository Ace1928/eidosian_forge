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
def test_reindex_method(self) -> None:
    ds = Dataset({'x': ('y', [10, 20]), 'y': [0, 1]})
    y = [-0.5, 0.5, 1.5]
    actual = ds.reindex(y=y, method='backfill')
    expected = Dataset({'x': ('y', [10, 20, np.nan]), 'y': y})
    assert_identical(expected, actual)
    actual = ds.reindex(y=y, method='backfill', tolerance=0.1)
    expected = Dataset({'x': ('y', 3 * [np.nan]), 'y': y})
    assert_identical(expected, actual)
    actual = ds.reindex(y=y, method='backfill', tolerance=[0.1, 0.5, 0.1])
    expected = Dataset({'x': ('y', [np.nan, 20, np.nan]), 'y': y})
    assert_identical(expected, actual)
    actual = ds.reindex(y=[0.1, 0.1, 1], tolerance=[0, 0.1, 0], method='nearest')
    expected = Dataset({'x': ('y', [np.nan, 10, 20]), 'y': [0.1, 0.1, 1]})
    assert_identical(expected, actual)
    actual = ds.reindex(y=y, method='pad')
    expected = Dataset({'x': ('y', [np.nan, 10, 20]), 'y': y})
    assert_identical(expected, actual)
    alt = Dataset({'y': y})
    actual = ds.reindex_like(alt, method='pad')
    assert_identical(expected, actual)