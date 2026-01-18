from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_tdi_add_overflow(self):
    with pytest.raises(OutOfBoundsDatetime, match='10155196800000000000'):
        pd.to_timedelta(106580, 'D') + Timestamp('2000')
    with pytest.raises(OutOfBoundsDatetime, match='10155196800000000000'):
        Timestamp('2000') + pd.to_timedelta(106580, 'D')
    _NaT = NaT._value + 1
    msg = 'Overflow in int64 addition'
    with pytest.raises(OverflowError, match=msg):
        pd.to_timedelta([106580], 'D') + Timestamp('2000')
    with pytest.raises(OverflowError, match=msg):
        Timestamp('2000') + pd.to_timedelta([106580], 'D')
    with pytest.raises(OverflowError, match=msg):
        pd.to_timedelta([_NaT]) - Timedelta('1 days')
    with pytest.raises(OverflowError, match=msg):
        pd.to_timedelta(['5 days', _NaT]) - Timedelta('1 days')
    with pytest.raises(OverflowError, match=msg):
        pd.to_timedelta([_NaT, '5 days', '1 hours']) - pd.to_timedelta(['7 seconds', _NaT, '4 hours'])
    exp = TimedeltaIndex([NaT])
    result = pd.to_timedelta([NaT]) - Timedelta('1 days')
    tm.assert_index_equal(result, exp)
    exp = TimedeltaIndex(['4 days', NaT])
    result = pd.to_timedelta(['5 days', NaT]) - Timedelta('1 days')
    tm.assert_index_equal(result, exp)
    exp = TimedeltaIndex([NaT, NaT, '5 hours'])
    result = pd.to_timedelta([NaT, '5 days', '1 hours']) + pd.to_timedelta(['7 seconds', NaT, '4 hours'])
    tm.assert_index_equal(result, exp)