import pytest
from pandas import IntervalIndex
import pandas._testing as tm
@pytest.mark.parametrize('closed', ['left', 'right', 'both'])
def test_pickle_round_trip_closed(self, closed):
    idx = IntervalIndex.from_tuples([(1, 2), (2, 3)], closed=closed)
    result = tm.round_trip_pickle(idx)
    tm.assert_index_equal(result, idx)