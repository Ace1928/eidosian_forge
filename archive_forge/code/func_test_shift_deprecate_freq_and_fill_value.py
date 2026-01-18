import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_deprecate_freq_and_fill_value(self, frame_or_series):
    obj = frame_or_series(np.random.default_rng(2).standard_normal(5), index=date_range('1/1/2000', periods=5, freq='h'))
    msg = "Passing a 'freq' together with a 'fill_value' silently ignores the fill_value"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        obj.shift(1, fill_value=1, freq='h')
    if frame_or_series is DataFrame:
        obj.columns = date_range('1/1/2000', periods=1, freq='h')
        with tm.assert_produces_warning(FutureWarning, match=msg):
            obj.shift(1, axis=1, fill_value=1, freq='h')