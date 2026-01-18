from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_coerce_strings_unit(self):
    arr = np.array([1, 2, 'error'], dtype=object)
    result = to_timedelta(arr, unit='ns', errors='coerce')
    expected = to_timedelta([1, 2, pd.NaT], unit='ns')
    tm.assert_index_equal(result, expected)