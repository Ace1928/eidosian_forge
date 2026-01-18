import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_tz_convert_naive(self, frame_or_series):
    rng = date_range('1/1/2011', periods=200, freq='D')
    ts = Series(1, index=rng)
    ts = frame_or_series(ts)
    with pytest.raises(TypeError, match='Cannot convert tz-naive'):
        ts.tz_convert('US/Eastern')