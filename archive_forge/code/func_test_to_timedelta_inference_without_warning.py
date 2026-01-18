from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_inference_without_warning(self):
    vals = ['00:00:01', pd.NaT]
    with tm.assert_produces_warning(None):
        result = to_timedelta(vals)
    expected = TimedeltaIndex([pd.Timedelta(seconds=1), pd.NaT])
    tm.assert_index_equal(result, expected)