from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('how', ['agg', 'apply'])
def test_size_as_str(how, axis):
    df = DataFrame({'A': [None, 2, 3], 'B': [1.0, np.nan, 3.0], 'C': ['foo', None, 'bar']})
    result = getattr(df, how)('size', axis=axis)
    if axis in (0, 'index'):
        expected = Series(df.shape[0], index=df.columns)
    else:
        expected = Series(df.shape[1], index=df.index)
    tm.assert_series_equal(result, expected)