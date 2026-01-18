from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_oob_non_nano(self):
    arr = np.array([pd.NaT._value + 1], dtype='timedelta64[m]')
    msg = 'Cannot convert -9223372036854775807 minutes to timedelta64\\[s\\] without overflow'
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        to_timedelta(arr)
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        TimedeltaIndex(arr)
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        TimedeltaArray._from_sequence(arr, dtype='m8[s]')