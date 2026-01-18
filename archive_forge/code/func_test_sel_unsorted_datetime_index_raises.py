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
def test_sel_unsorted_datetime_index_raises(self) -> None:
    index = PandasIndex(pd.to_datetime(['2001', '2000', '2002']), 'x')
    with pytest.raises(KeyError):
        index.sel({'x': slice('2001', '2002')})