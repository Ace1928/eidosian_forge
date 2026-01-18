from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_repr_of_dataarray(dataarray: xr.DataArray) -> None:
    formatted = fh.array_repr(dataarray)
    assert 'dim_0' in formatted
    assert formatted.count("class='xr-array-in' type='checkbox' checked>") == 1
    assert formatted.count("class='xr-section-summary-in' type='checkbox' disabled >") == 3
    with xr.set_options(display_expand_data=False):
        formatted = fh.array_repr(dataarray)
        assert 'dim_0' in formatted
        assert formatted.count("class='xr-array-in' type='checkbox' checked>") == 0
        assert formatted.count("class='xr-section-summary-in' type='checkbox' disabled >") == 3