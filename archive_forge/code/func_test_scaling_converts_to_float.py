from __future__ import annotations
from contextlib import suppress
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding import variables
from xarray.conventions import decode_cf_variable, encode_cf_variable
from xarray.tests import assert_allclose, assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('dtype', 'u1 u2 i1 i2 f2 f4'.split())
@pytest.mark.parametrize('dtype2', 'f4 f8'.split())
def test_scaling_converts_to_float(dtype: str, dtype2: str) -> None:
    dt = np.dtype(dtype2)
    original = xr.Variable(('x',), np.arange(10, dtype=dtype), encoding=dict(scale_factor=dt.type(10)))
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == dt
    roundtripped = coder.decode(encoded)
    assert_identical(original, roundtripped)
    assert roundtripped.dtype == dt