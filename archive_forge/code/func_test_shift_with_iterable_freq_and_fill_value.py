import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_with_iterable_freq_and_fill_value(self):
    df = DataFrame(np.random.default_rng(2).standard_normal(5), index=date_range('1/1/2000', periods=5, freq='h'))
    tm.assert_frame_equal(df.shift([1], fill_value=1).rename(columns=lambda x: int(x[0])), df.shift(1, fill_value=1))
    tm.assert_frame_equal(df.shift([1], freq='h').rename(columns=lambda x: int(x[0])), df.shift(1, freq='h'))
    msg = "Passing a 'freq' together with a 'fill_value' silently ignores the fill_value"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.shift([1, 2], fill_value=1, freq='h')