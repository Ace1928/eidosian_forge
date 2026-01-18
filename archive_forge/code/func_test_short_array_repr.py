from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
def test_short_array_repr() -> None:
    cases = [np.random.randn(500), np.random.randn(20, 20), np.random.randn(5, 10, 15), np.random.randn(5, 10, 15, 3), np.random.randn(100, 5, 1)]
    for array in cases:
        num_lines = formatting.short_array_repr(array).count('\n') + 1
        assert num_lines < 30
    array2 = np.arange(100)
    assert '...' not in formatting.short_array_repr(array2)
    with xr.set_options(display_values_threshold=10):
        assert '...' in formatting.short_array_repr(array2)