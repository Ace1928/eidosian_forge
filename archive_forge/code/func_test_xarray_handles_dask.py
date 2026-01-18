from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_allclose, assert_array_equal, mock
from xarray.tests import assert_identical as assert_identical_
def test_xarray_handles_dask():
    da = pytest.importorskip('dask.array')
    x = xr.DataArray(np.ones((2, 2)), dims=['x', 'y'])
    y = da.ones((2, 2), chunks=(2, 2))
    result = np.add(x, y)
    assert result.chunks == ((2,), (2,))
    assert isinstance(result, xr.DataArray)