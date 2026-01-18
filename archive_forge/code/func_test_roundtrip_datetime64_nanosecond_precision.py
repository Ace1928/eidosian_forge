from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@pytest.mark.parametrize('timestr, timeunit, dtype, fill_value, use_encoding', [('1677-09-21T00:12:43.145224193', 'ns', np.int64, 20, True), ('1970-09-21T00:12:44.145224808', 'ns', np.float64, 1e+30, True), ('1677-09-21T00:12:43.145225216', 'ns', np.float64, -9.223372036854776e+18, True), ('1677-09-21T00:12:43.145224193', 'ns', np.int64, None, False), ('1677-09-21T00:12:43.145225', 'us', np.int64, None, False), ('1970-01-01T00:00:01.000001', 'us', np.int64, None, False), ('1677-09-21T00:21:52.901038080', 'ns', np.float32, 20.0, True)])
def test_roundtrip_datetime64_nanosecond_precision(timestr: str, timeunit: str, dtype: np.typing.DTypeLike, fill_value: int | float | None, use_encoding: bool) -> None:
    time = np.datetime64(timestr, timeunit)
    times = [np.datetime64('1970-01-01T00:00:00', timeunit), np.datetime64('NaT'), time]
    if use_encoding:
        encoding = dict(dtype=dtype, _FillValue=fill_value)
    else:
        encoding = {}
    var = Variable(['time'], times, encoding=encoding)
    assert var.dtype == np.dtype('<M8[ns]')
    encoded_var = conventions.encode_cf_variable(var)
    assert encoded_var.attrs['units'] == f'{_numpy_to_netcdf_timeunit(timeunit)} since 1970-01-01 00:00:00'
    assert encoded_var.attrs['calendar'] == 'proleptic_gregorian'
    assert encoded_var.data.dtype == dtype
    decoded_var = conventions.decode_cf_variable('foo', encoded_var)
    assert decoded_var.dtype == np.dtype('<M8[ns]')
    assert decoded_var.encoding['units'] == f'{_numpy_to_netcdf_timeunit(timeunit)} since 1970-01-01 00:00:00'
    assert decoded_var.encoding['dtype'] == dtype
    assert decoded_var.encoding['calendar'] == 'proleptic_gregorian'
    assert_identical(var, decoded_var)