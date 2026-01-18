from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_false_index_name():
    result_series = Series(data=range(5, 10), index=range(5))
    result_series.index.name = False
    result_series.reset_index()
    expected_series = Series(range(5, 10), RangeIndex(range(5), name=False))
    tm.assert_series_equal(result_series, expected_series)
    result_frame = DataFrame(data=range(5, 10), index=range(5))
    result_frame.index.name = False
    result_frame.reset_index()
    expected_frame = DataFrame(range(5, 10), RangeIndex(range(5), name=False))
    tm.assert_frame_equal(result_frame, expected_frame)