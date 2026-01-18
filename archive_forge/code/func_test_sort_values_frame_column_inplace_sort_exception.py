import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_frame_column_inplace_sort_exception(self, float_frame, using_copy_on_write):
    s = float_frame['A']
    float_frame_orig = float_frame.copy()
    if using_copy_on_write:
        s.sort_values(inplace=True)
        tm.assert_series_equal(s, float_frame_orig['A'].sort_values())
        tm.assert_frame_equal(float_frame, float_frame_orig)
    else:
        with pytest.raises(ValueError, match='This Series is a view'):
            s.sort_values(inplace=True)
    cp = s.copy()
    cp.sort_values()