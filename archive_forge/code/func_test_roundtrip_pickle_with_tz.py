import pytest
from pandas import (
import pandas._testing as tm
def test_roundtrip_pickle_with_tz(self):
    index = date_range('20130101', periods=3, tz='US/Eastern', name='foo')
    unpickled = tm.round_trip_pickle(index)
    tm.assert_index_equal(index, unpickled)