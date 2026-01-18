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
def test_to_pandas_index(self) -> None:
    pd_idx = pd.Index([1, 2, 3], name='foo')
    index = PandasIndex(pd_idx, 'x')
    assert index.to_pandas_index() is index.index