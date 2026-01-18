from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_inplace_and_drop_ignore_name(self):
    ser = Series(range(2), name='old')
    ser.reset_index(name='new', drop=True, inplace=True)
    expected = Series(range(2), name='old')
    tm.assert_series_equal(ser, expected)