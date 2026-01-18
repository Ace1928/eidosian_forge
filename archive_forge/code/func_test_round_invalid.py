import pytest
from pandas._libs.tslibs import to_offset
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq, error_msg', [('YE', '<YearEnd: month=12> is a non-fixed frequency'), ('ME', '<MonthEnd> is a non-fixed frequency'), ('foobar', 'Invalid frequency: foobar')])
def test_round_invalid(self, freq, error_msg):
    dti = date_range('20130101 09:10:11', periods=5)
    dti = dti.tz_localize('UTC').tz_convert('US/Eastern')
    with pytest.raises(ValueError, match=error_msg):
        dti.round(freq)