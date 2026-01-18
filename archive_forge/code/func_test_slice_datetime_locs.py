from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('box', [datetime, Timestamp])
def test_slice_datetime_locs(self, box, tz_aware_fixture):
    tz = tz_aware_fixture
    index = DatetimeIndex(['2010-01-01', '2010-01-03']).tz_localize(tz)
    key = box(2010, 1, 1)
    if tz is not None:
        with pytest.raises(TypeError, match='Cannot compare tz-naive'):
            index.slice_locs(key, box(2010, 1, 2))
    else:
        result = index.slice_locs(key, box(2010, 1, 2))
        expected = (0, 1)
        assert result == expected