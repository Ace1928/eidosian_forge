from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
def test_sel(self) -> None:
    index = PandasMultiIndex(pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('one', 'two')), 'x')
    actual = index.sel({'x': slice(('a', 1), ('b', 2))})
    expected_dim_indexers = {'x': slice(0, 4)}
    assert actual.dim_indexers == expected_dim_indexers
    with pytest.raises(KeyError, match='not all values found'):
        index.sel({'x': [0]})
    with pytest.raises(KeyError):
        index.sel({'x': 0})
    with pytest.raises(ValueError, match='cannot provide labels for both.*'):
        index.sel({'one': 0, 'x': 'a'})
    with pytest.raises(ValueError, match="multi-index level names \\('three',\\) not found in indexes"):
        index.sel({'x': {'three': 0}})
    with pytest.raises(IndexError):
        index.sel({'x': (slice(None), 1, 'no_level')})