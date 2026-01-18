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
def test_reorder_levels(self) -> None:
    ds = create_test_multiindex()
    mindex = ds['x'].to_index()
    midx = mindex.reorder_levels(['level_2', 'level_1'])
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    expected = Dataset({}, coords=midx_coords)
    ds['level_1'].attrs['foo'] = 'bar'
    expected['level_1'].attrs['foo'] = 'bar'
    reindexed = ds.reorder_levels(x=['level_2', 'level_1'])
    assert_identical(reindexed, expected)
    ds = Dataset({}, coords={'x': [1, 2]})
    with pytest.raises(ValueError, match='has no MultiIndex'):
        ds.reorder_levels(x=['level_1', 'level_2'])