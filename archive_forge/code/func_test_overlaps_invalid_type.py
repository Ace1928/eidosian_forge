import pytest
from pandas import (
@pytest.mark.parametrize('other', [10, True, 'foo', Timedelta('1 day'), Timestamp('2018-01-01')], ids=lambda x: type(x).__name__)
def test_overlaps_invalid_type(self, other):
    interval = Interval(0, 1)
    msg = f'`other` must be an Interval, got {type(other).__name__}'
    with pytest.raises(TypeError, match=msg):
        interval.overlaps(other)