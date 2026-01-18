from __future__ import annotations
import datetime
from typing import TypedDict
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core.pdcompat import _convert_base_to_offset
from xarray.core.resample_cftime import CFTimeGrouper
@pytest.mark.parametrize('loffset', ['MS', '12h', datetime.timedelta(hours=-12)])
def test_resample_loffset_cftimeindex(loffset) -> None:
    datetimeindex = pd.date_range('2000-01-01', freq='6h', periods=10)
    da_datetimeindex = xr.DataArray(np.arange(10), [('time', datetimeindex)])
    cftimeindex = xr.cftime_range('2000-01-01', freq='6h', periods=10)
    da_cftimeindex = xr.DataArray(np.arange(10), [('time', cftimeindex)])
    with pytest.warns(FutureWarning, match='`loffset` parameter'):
        result = da_cftimeindex.resample(time='24h', loffset=loffset).mean()
        expected = da_datetimeindex.resample(time='24h', loffset=loffset).mean()
    result['time'] = result.xindexes['time'].to_pandas_index().to_datetimeindex()
    xr.testing.assert_identical(result, expected)