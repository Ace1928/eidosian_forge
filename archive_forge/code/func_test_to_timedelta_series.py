from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_series(self):
    expected = Series([timedelta(days=1), timedelta(days=1, seconds=1)])
    result = to_timedelta(Series(['1d', '1days 00:00:01']))
    tm.assert_series_equal(result, expected)