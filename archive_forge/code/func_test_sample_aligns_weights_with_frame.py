import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_aligns_weights_with_frame(self):
    df = DataFrame({'col1': [5, 6, 7], 'col2': ['a', 'b', 'c']}, index=[9, 5, 3])
    ser = Series([1, 0, 0], index=[3, 5, 9])
    tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser))
    ser2 = Series([0.001, 0, 10000], index=[3, 5, 10])
    tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser2))
    ser3 = Series([0.01, 0], index=[3, 5])
    tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser3))
    ser4 = Series([1, 0], index=[1, 2])
    with pytest.raises(ValueError, match='Invalid weights: weights sum to zero'):
        df.sample(1, weights=ser4)