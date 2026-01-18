import pytest
from pandas import TimedeltaIndex
from pandas.tseries.offsets import (
def test_with_freq_empty_requires_tick(self):
    idx = TimedeltaIndex([])
    off = MonthEnd(1)
    msg = 'TimedeltaArray/Index freq must be a Tick'
    with pytest.raises(TypeError, match=msg):
        idx._with_freq(off)
    with pytest.raises(TypeError, match=msg):
        idx._data._with_freq(off)