from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('value', [pd.Period('2020-01'), pd.Interval(0, 5)])
def test_replace_ea_ignore_float(self, frame_or_series, value):
    obj = DataFrame({'Per': [value] * 3})
    obj = tm.get_obj(obj, frame_or_series)
    expected = obj.copy()
    result = obj.replace(1.0, 0.0)
    tm.assert_equal(expected, result)