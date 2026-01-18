import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
def test_searchsorted_datetime64_scalar(self):
    ser = Series(date_range('20120101', periods=10, freq='2D'))
    val = Timestamp('20120102')
    res = ser.searchsorted(val)
    assert is_scalar(res)
    assert res == 1