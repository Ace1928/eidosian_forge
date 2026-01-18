import pytest
from pandas._libs.tslibs import to_offset
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_round_daily(self):
    dti = date_range('20130101 09:10:11', periods=5)
    result = dti.round('D')
    expected = date_range('20130101', periods=5)
    tm.assert_index_equal(result, expected)
    dti = dti.tz_localize('UTC').tz_convert('US/Eastern')
    result = dti.round('D')
    expected = date_range('20130101', periods=5).tz_localize('US/Eastern')
    tm.assert_index_equal(result, expected)
    result = dti.round('s')
    tm.assert_index_equal(result, dti)