import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_datetimelike_attribute_access_invalid(self):
    ser = Series({'year': 2000, 'month': 1, 'day': 10})
    msg = "'Series' object has no attribute 'weekday'"
    with pytest.raises(AttributeError, match=msg):
        ser.weekday