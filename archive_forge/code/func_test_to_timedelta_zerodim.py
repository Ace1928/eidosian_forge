from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_zerodim(self, fixed_now_ts):
    dt64 = fixed_now_ts.to_datetime64()
    arg = np.array(dt64)
    msg = 'Value must be Timedelta, string, integer, float, timedelta or convertible, not datetime64'
    with pytest.raises(ValueError, match=msg):
        to_timedelta(arg)
    arg2 = arg.view('m8[ns]')
    result = to_timedelta(arg2)
    assert isinstance(result, pd.Timedelta)
    assert result._value == dt64.view('i8')