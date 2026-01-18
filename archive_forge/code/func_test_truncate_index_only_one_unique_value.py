import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_truncate_index_only_one_unique_value(self, frame_or_series):
    obj = Series(0, index=date_range('2021-06-30', '2021-06-30')).repeat(5)
    if frame_or_series is DataFrame:
        obj = obj.to_frame(name='a')
    truncated = obj.truncate('2021-06-28', '2021-07-01')
    tm.assert_equal(truncated, obj)