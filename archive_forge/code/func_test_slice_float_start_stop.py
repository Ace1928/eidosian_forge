import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_slice_float_start_stop(self, series_with_interval_index):
    ser = series_with_interval_index.copy()
    msg = 'label-based slicing with step!=1 is not supported for IntervalIndex'
    with pytest.raises(ValueError, match=msg):
        ser[1.5:9.5:2]