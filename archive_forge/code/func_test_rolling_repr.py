from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('da', (1,), indirect=True)
def test_rolling_repr(self, da) -> None:
    rolling_obj = da.rolling(time=7)
    assert repr(rolling_obj) == 'DataArrayRolling [time->7]'
    rolling_obj = da.rolling(time=7, center=True)
    assert repr(rolling_obj) == 'DataArrayRolling [time->7(center)]'
    rolling_obj = da.rolling(time=7, x=3, center=True)
    assert repr(rolling_obj) == 'DataArrayRolling [time->7(center),x->3(center)]'