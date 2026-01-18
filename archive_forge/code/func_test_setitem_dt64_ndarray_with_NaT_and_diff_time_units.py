from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_dt64_ndarray_with_NaT_and_diff_time_units(self):
    data_ns = np.array([1, 'nat'], dtype='datetime64[ns]')
    result = Series(data_ns).to_frame()
    result['new'] = data_ns
    expected = DataFrame({0: [1, None], 'new': [1, None]}, dtype='datetime64[ns]')
    tm.assert_frame_equal(result, expected)
    data_s = np.array([1, 'nat'], dtype='datetime64[s]')
    result['new'] = data_s
    tm.assert_series_equal(result[0], expected[0])
    tm.assert_numpy_array_equal(result['new'].to_numpy(), data_s)