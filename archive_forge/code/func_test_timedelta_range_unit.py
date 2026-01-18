import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_timedelta_range_unit(self):
    tdi = timedelta_range('0 Days', periods=10, freq='100000D', unit='s')
    exp_arr = (np.arange(10, dtype='i8') * 100000).view('m8[D]').astype('m8[s]')
    tm.assert_numpy_array_equal(tdi.to_numpy(), exp_arr)