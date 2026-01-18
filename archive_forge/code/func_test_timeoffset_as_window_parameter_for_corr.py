from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_timeoffset_as_window_parameter_for_corr(unit):
    dti = DatetimeIndex([Timestamp('20130101 09:00:00'), Timestamp('20130102 09:00:02'), Timestamp('20130103 09:00:03'), Timestamp('20130105 09:00:05'), Timestamp('20130106 09:00:06')]).as_unit(unit)
    mi = MultiIndex.from_product([dti, ['B', 'A']])
    exp = DataFrame({'B': [np.nan, np.nan, 0.9999999999999998, -1.0, 1.0, -0.3273268353539892, 0.9999999999999998, 1.0, 0.9999999999999998, 1.0], 'A': [np.nan, np.nan, -1.0, 1.0000000000000002, -0.3273268353539892, 0.9999999999999966, 1.0, 1.0000000000000002, 1.0, 1.0000000000000002]}, index=mi)
    df = DataFrame({'B': [0, 1, 2, 4, 3], 'A': [7, 4, 6, 9, 3]}, index=dti)
    res = df.rolling(window='3d').corr()
    tm.assert_frame_equal(exp, res)