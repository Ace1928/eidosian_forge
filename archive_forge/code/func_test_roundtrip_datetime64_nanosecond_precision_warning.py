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
def test_roundtrip_datetime64_nanosecond_precision_warning() -> None:
    times = [np.datetime64('1970-01-01T00:01:00', 'ns'), np.datetime64('NaT'), np.datetime64('1970-01-02T00:01:00', 'ns')]
    units = 'days since 1970-01-10T01:01:00'
    needed_units = 'hours'
    new_units = f'{needed_units} since 1970-01-10T01:01:00'
    encoding = dict(dtype=None, _FillValue=20, units=units)
    var = Variable(['time'], times, encoding=encoding)
    with pytest.warns(UserWarning, match=f'Resolution of {needed_units!r} needed.'):
        encoded_var = conventions.encode_cf_variable(var)
    assert encoded_var.dtype == np.float64
    assert encoded_var.attrs['units'] == units
    assert encoded_var.attrs['_FillValue'] == 20.0
    decoded_var = conventions.decode_cf_variable('foo', encoded_var)
    assert_identical(var, decoded_var)
    encoding = dict(dtype='int64', _FillValue=20, units=units)
    var = Variable(['time'], times, encoding=encoding)
    with pytest.warns(UserWarning, match=f'Serializing with units {new_units!r} instead.'):
        encoded_var = conventions.encode_cf_variable(var)
    assert encoded_var.dtype == np.int64
    assert encoded_var.attrs['units'] == new_units
    assert encoded_var.attrs['_FillValue'] == 20
    decoded_var = conventions.decode_cf_variable('foo', encoded_var)
    assert_identical(var, decoded_var)
    encoding = dict(dtype='float64', _FillValue=20, units=units)
    var = Variable(['time'], times, encoding=encoding)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        encoded_var = conventions.encode_cf_variable(var)
    assert encoded_var.dtype == np.float64
    assert encoded_var.attrs['units'] == units
    assert encoded_var.attrs['_FillValue'] == 20.0
    decoded_var = conventions.decode_cf_variable('foo', encoded_var)
    assert_identical(var, decoded_var)
    encoding = dict(dtype='int64', _FillValue=20, units=new_units)
    var = Variable(['time'], times, encoding=encoding)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        encoded_var = conventions.encode_cf_variable(var)
    assert encoded_var.dtype == np.int64
    assert encoded_var.attrs['units'] == new_units
    assert encoded_var.attrs['_FillValue'] == 20
    decoded_var = conventions.decode_cf_variable('foo', encoded_var)
    assert_identical(var, decoded_var)