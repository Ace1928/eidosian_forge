import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_td64_summation_overflow():
    ser = Series(pd.date_range('20130101', periods=100000, freq='h'))
    ser[0] += pd.Timedelta('1s 1ms')
    result = (ser - ser.min()).mean()
    expected = pd.Timedelta((pd.TimedeltaIndex(ser - ser.min()).asi8 / len(ser)).sum())
    assert np.allclose(result._value / 1000, expected._value / 1000)
    msg = 'overflow in timedelta operation'
    with pytest.raises(ValueError, match=msg):
        (ser - ser.min()).sum()
    s1 = ser[0:10000]
    with pytest.raises(ValueError, match=msg):
        (s1 - s1.min()).sum()
    s2 = ser[0:1000]
    (s2 - s2.min()).sum()