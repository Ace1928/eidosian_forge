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
def test_unstack(self) -> None:
    pd_midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2, 3]], names=['one', 'two'])
    index = PandasMultiIndex(pd_midx, 'x')
    new_indexes, new_pd_idx = index.unstack()
    assert list(new_indexes) == ['one', 'two']
    assert new_indexes['one'].equals(PandasIndex(['a', 'b'], 'one'))
    assert new_indexes['two'].equals(PandasIndex([1, 2, 3], 'two'))
    assert new_pd_idx.equals(pd_midx)