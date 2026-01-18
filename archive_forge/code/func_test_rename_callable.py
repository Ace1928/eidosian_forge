from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_callable(self):
    ser = Series(range(1, 6), index=Index(range(2, 7), name='IntIndex'))
    result = ser.rename(str)
    expected = ser.rename(lambda i: str(i))
    tm.assert_series_equal(result, expected)
    assert result.name == expected.name