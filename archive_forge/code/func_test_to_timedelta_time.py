from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_time(self):
    msg = 'Value must be Timedelta, string, integer, float, timedelta or convertible'
    with pytest.raises(ValueError, match=msg):
        to_timedelta(time(second=1))
    assert to_timedelta(time(second=1), errors='coerce') is pd.NaT