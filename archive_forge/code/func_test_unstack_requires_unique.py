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
def test_unstack_requires_unique(self) -> None:
    pd_midx = pd.MultiIndex.from_product([['a', 'a'], [1, 2]], names=['one', 'two'])
    index = PandasMultiIndex(pd_midx, 'x')
    with pytest.raises(ValueError, match='Cannot unstack MultiIndex containing duplicates'):
        index.unstack()