import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_period_index_frame_shift_with_freq(self, frame_or_series):
    ps = DataFrame(range(4), index=pd.period_range('2020-01-01', periods=4))
    ps = tm.get_obj(ps, frame_or_series)
    shifted = ps.shift(1, freq='infer')
    unshifted = shifted.shift(-1, freq='infer')
    tm.assert_equal(unshifted, ps)
    shifted2 = ps.shift(freq='D')
    tm.assert_equal(shifted, shifted2)
    shifted3 = ps.shift(freq=offsets.Day())
    tm.assert_equal(shifted, shifted3)