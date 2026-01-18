import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_xarray_coerce_unit():
    xr = pytest.importorskip('xarray')
    arr = xr.DataArray([1, 2, 3])
    result = pd.to_datetime(arr, unit='ns')
    expected = DatetimeIndex(['1970-01-01 00:00:00.000000001', '1970-01-01 00:00:00.000000002', '1970-01-01 00:00:00.000000003'], dtype='datetime64[ns]', freq=None)
    tm.assert_index_equal(result, expected)