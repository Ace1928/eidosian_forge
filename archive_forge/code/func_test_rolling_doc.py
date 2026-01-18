from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
def test_rolling_doc(self, da) -> None:
    rolling_obj = da.rolling(time=7)
    assert '`mean`' in rolling_obj.mean.__doc__