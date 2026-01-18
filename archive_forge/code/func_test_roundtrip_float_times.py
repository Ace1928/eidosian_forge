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
def test_roundtrip_float_times() -> None:
    fill_value = 20.0
    times = [np.datetime64('1970-01-01 00:00:00', 'ns'), np.datetime64('1970-01-01 06:00:00', 'ns'), np.datetime64('NaT', 'ns')]
    units = 'days since 1960-01-01'
    var = Variable(['time'], times, encoding=dict(dtype=np.float64, _FillValue=fill_value, units=units))
    encoded_var = conventions.encode_cf_variable(var)
    np.testing.assert_array_equal(encoded_var, np.array([3653, 3653.25, 20.0]))
    assert encoded_var.attrs['units'] == units
    assert encoded_var.attrs['_FillValue'] == fill_value
    decoded_var = conventions.decode_cf_variable('foo', encoded_var)
    assert_identical(var, decoded_var)
    assert decoded_var.encoding['units'] == units
    assert decoded_var.encoding['_FillValue'] == fill_value