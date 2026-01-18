from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_value_counts_nat(self):
    td = Series([np.timedelta64(10000), NaT], dtype='timedelta64[ns]')
    dt = to_datetime(['NaT', '2014-01-01'])
    msg = 'pandas.value_counts is deprecated'
    for ser in [td, dt]:
        with tm.assert_produces_warning(FutureWarning, match=msg):
            vc = algos.value_counts(ser)
            vc_with_na = algos.value_counts(ser, dropna=False)
        assert len(vc) == 1
        assert len(vc_with_na) == 2
    exp_dt = Series({Timestamp('2014-01-01 00:00:00'): 1}, name='count')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result_dt = algos.value_counts(dt)
    tm.assert_series_equal(result_dt, exp_dt)
    exp_td = Series({np.timedelta64(10000): 1}, name='count')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result_td = algos.value_counts(td)
    tm.assert_series_equal(result_td, exp_td)