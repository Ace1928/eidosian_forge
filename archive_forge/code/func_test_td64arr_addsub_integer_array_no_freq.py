from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_addsub_integer_array_no_freq(self, box_with_array):
    box = box_with_array
    xbox = np.ndarray if box is pd.array else box
    tdi = TimedeltaIndex(['1 Day', 'NaT', '3 Hours'])
    tdarr = tm.box_expected(tdi, box)
    other = tm.box_expected([14, -1, 16], xbox)
    msg = 'Addition/subtraction of integers'
    assert_invalid_addsub_type(tdarr, other, msg)