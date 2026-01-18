from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slice_requires_monotonicity(self):
    ser = Series(np.arange(10), date_range('2014-01-01', periods=10))
    nonmonotonic = ser.iloc[[3, 5, 4]]
    timestamp = Timestamp('2014-01-10')
    with pytest.raises(KeyError, match='Value based partial slicing on non-monotonic'):
        nonmonotonic['2014-01-10':]
    with pytest.raises(KeyError, match="Timestamp\\('2014-01-10 00:00:00'\\)"):
        nonmonotonic[timestamp:]
    with pytest.raises(KeyError, match='Value based partial slicing on non-monotonic'):
        nonmonotonic.loc['2014-01-10':]
    with pytest.raises(KeyError, match="Timestamp\\('2014-01-10 00:00:00'\\)"):
        nonmonotonic.loc[timestamp:]