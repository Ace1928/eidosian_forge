from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_sub_timedeltalike(self, two_hours, box_with_array):
    box = box_with_array
    rng = timedelta_range('1 days', '10 days')
    expected = timedelta_range('0 days 22:00:00', '9 days 22:00:00')
    rng = tm.box_expected(rng, box)
    expected = tm.box_expected(expected, box)
    result = rng - two_hours
    tm.assert_equal(result, expected)
    result = two_hours - rng
    tm.assert_equal(result, -expected)