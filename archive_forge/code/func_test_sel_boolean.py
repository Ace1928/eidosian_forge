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
def test_sel_boolean(self) -> None:
    index = PandasIndex(pd.Index([0.0, 2.0, 1.0, 3.0]), 'x')
    actual = index.sel({'x': [False, True, False, True]})
    expected_dim_indexers = {'x': [False, True, False, True]}
    np.testing.assert_array_equal(actual.dim_indexers['x'], expected_dim_indexers['x'])