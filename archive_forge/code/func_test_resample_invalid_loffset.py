from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
def test_resample_invalid_loffset(self) -> None:
    times = pd.date_range('2000-01-01', freq='6h', periods=10)
    array = DataArray(np.arange(10), [('time', times)])
    with pytest.warns(FutureWarning, match='Following pandas, the `loffset` parameter'):
        with pytest.raises(ValueError, match='`loffset` must be'):
            array.resample(time='24h', loffset=1).mean()