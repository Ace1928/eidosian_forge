from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_chained_setfault(self, using_copy_on_write):
    data = ['right', 'left', 'left', 'left', 'right', 'left', 'timeout']
    mdata = ['right', 'left', 'left', 'left', 'right', 'left', 'none']
    df = DataFrame({'response': np.array(data)})
    mask = df.response == 'timeout'
    with tm.raises_chained_assignment_error():
        df.response[mask] = 'none'
    if using_copy_on_write:
        tm.assert_frame_equal(df, DataFrame({'response': data}))
    else:
        tm.assert_frame_equal(df, DataFrame({'response': mdata}))
    recarray = np.rec.fromarrays([data], names=['response'])
    df = DataFrame(recarray)
    mask = df.response == 'timeout'
    with tm.raises_chained_assignment_error():
        df.response[mask] = 'none'
    if using_copy_on_write:
        tm.assert_frame_equal(df, DataFrame({'response': data}))
    else:
        tm.assert_frame_equal(df, DataFrame({'response': mdata}))
    df = DataFrame({'response': data, 'response1': data})
    df_original = df.copy()
    mask = df.response == 'timeout'
    with tm.raises_chained_assignment_error():
        df.response[mask] = 'none'
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_original)
    else:
        tm.assert_frame_equal(df, DataFrame({'response': mdata, 'response1': data}))
    expected = DataFrame({'A': [np.nan, 'bar', 'bah', 'foo', 'bar']})
    df = DataFrame({'A': np.array(['foo', 'bar', 'bah', 'foo', 'bar'])})
    with tm.raises_chained_assignment_error():
        df['A'].iloc[0] = np.nan
    if using_copy_on_write:
        expected = DataFrame({'A': ['foo', 'bar', 'bah', 'foo', 'bar']})
    else:
        expected = DataFrame({'A': [np.nan, 'bar', 'bah', 'foo', 'bar']})
    result = df.head()
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'A': np.array(['foo', 'bar', 'bah', 'foo', 'bar'])})
    with tm.raises_chained_assignment_error():
        df.A.iloc[0] = np.nan
    result = df.head()
    tm.assert_frame_equal(result, expected)