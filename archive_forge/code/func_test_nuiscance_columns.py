from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_nuiscance_columns():
    df = DataFrame({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0], 'C': ['foo', 'bar', 'baz'], 'D': date_range('20130101', periods=3)})
    result = df.agg('min')
    expected = Series([1, 1.0, 'bar', Timestamp('20130101')], index=df.columns)
    tm.assert_series_equal(result, expected)
    result = df.agg(['min'])
    expected = DataFrame([[1, 1.0, 'bar', Timestamp('20130101').as_unit('ns')]], index=['min'], columns=df.columns)
    tm.assert_frame_equal(result, expected)
    msg = 'does not support reduction'
    with pytest.raises(TypeError, match=msg):
        df.agg('sum')
    result = df[['A', 'B', 'C']].agg('sum')
    expected = Series([6, 6.0, 'foobarbaz'], index=['A', 'B', 'C'])
    tm.assert_series_equal(result, expected)
    msg = 'does not support reduction'
    with pytest.raises(TypeError, match=msg):
        df.agg(['sum'])