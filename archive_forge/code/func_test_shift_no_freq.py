import pytest
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import TimedeltaIndex
import pandas._testing as tm
def test_shift_no_freq(self):
    tdi = TimedeltaIndex(['1 days 01:00:00', '2 days 01:00:00'], freq=None)
    with pytest.raises(NullFrequencyError, match='Cannot shift with no freq'):
        tdi.shift(2)