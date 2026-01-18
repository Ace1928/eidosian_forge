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
def test_resample_bad_resample_dim(self) -> None:
    times = pd.date_range('2000-01-01', freq='6h', periods=10)
    array = DataArray(np.arange(10), [('__resample_dim__', times)])
    with pytest.raises(ValueError, match='Proxy resampling dimension'):
        array.resample(**{'__resample_dim__': '1D'}).first()