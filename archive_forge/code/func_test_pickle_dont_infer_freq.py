import pytest
from pandas import (
import pandas._testing as tm
def test_pickle_dont_infer_freq(self):
    idx = date_range('1750-1-1', '2050-1-1', freq='7D')
    idx_p = tm.round_trip_pickle(idx)
    tm.assert_index_equal(idx, idx_p)