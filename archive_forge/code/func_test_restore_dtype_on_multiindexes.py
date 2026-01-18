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
@pytest.mark.parametrize('dtype', ['int32', 'float32'])
def test_restore_dtype_on_multiindexes(dtype: str) -> None:
    foo = xr.Dataset(coords={'bar': ('bar', np.array([0, 1], dtype=dtype))})
    foo = foo.stack(baz=('bar',))
    assert str(foo['bar'].values.dtype) == dtype