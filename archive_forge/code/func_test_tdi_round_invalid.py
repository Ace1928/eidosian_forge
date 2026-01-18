import numpy as np
import pytest
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq,msg', [('YE', '<YearEnd: month=12> is a non-fixed frequency'), ('ME', '<MonthEnd> is a non-fixed frequency'), ('foobar', 'Invalid frequency: foobar')])
def test_tdi_round_invalid(self, freq, msg):
    t1 = timedelta_range('1 days', periods=3, freq='1 min 2 s 3 us')
    with pytest.raises(ValueError, match=msg):
        t1.round(freq)
    with pytest.raises(ValueError, match=msg):
        t1._data.round(freq)