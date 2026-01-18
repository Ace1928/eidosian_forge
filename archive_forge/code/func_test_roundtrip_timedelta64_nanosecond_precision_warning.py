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
def test_roundtrip_timedelta64_nanosecond_precision_warning() -> None:
    one_day = np.timedelta64(1, 'D')
    nat = np.timedelta64('nat', 'ns')
    timedelta_values = (np.arange(5) * one_day).astype('timedelta64[ns]')
    timedelta_values[2] = nat
    timedelta_values[4] = np.timedelta64(12, 'h').astype('timedelta64[ns]')
    units = 'days'
    needed_units = 'hours'
    wmsg = f"Timedeltas can't be serialized faithfully with requested units {units!r}. Serializing with units {needed_units!r} instead."
    encoding = dict(dtype=np.int64, _FillValue=20, units=units)
    var = Variable(['time'], timedelta_values, encoding=encoding)
    with pytest.warns(UserWarning, match=wmsg):
        encoded_var = conventions.encode_cf_variable(var)
    assert encoded_var.dtype == np.int64
    assert encoded_var.attrs['units'] == needed_units
    assert encoded_var.attrs['_FillValue'] == 20
    decoded_var = conventions.decode_cf_variable('foo', encoded_var)
    assert_identical(var, decoded_var)
    assert decoded_var.encoding['dtype'] == np.int64