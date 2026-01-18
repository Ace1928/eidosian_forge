import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_include_using_list_like(self):
    df = DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6).astype('u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.Categorical(list('abc')), 'g': pd.date_range('20130101', periods=3), 'h': pd.date_range('20130101', periods=3, tz='US/Eastern'), 'i': pd.date_range('20130101', periods=3, tz='CET'), 'j': pd.period_range('2013-01', periods=3, freq='M'), 'k': pd.timedelta_range('1 day', periods=3)})
    ri = df.select_dtypes(include=[np.number])
    ei = df[['b', 'c', 'd', 'k']]
    tm.assert_frame_equal(ri, ei)
    ri = df.select_dtypes(include=[np.number], exclude=['timedelta'])
    ei = df[['b', 'c', 'd']]
    tm.assert_frame_equal(ri, ei)
    ri = df.select_dtypes(include=[np.number, 'category'], exclude=['timedelta'])
    ei = df[['b', 'c', 'd', 'f']]
    tm.assert_frame_equal(ri, ei)
    ri = df.select_dtypes(include=['datetime'])
    ei = df[['g']]
    tm.assert_frame_equal(ri, ei)
    ri = df.select_dtypes(include=['datetime64'])
    ei = df[['g']]
    tm.assert_frame_equal(ri, ei)
    ri = df.select_dtypes(include=['datetimetz'])
    ei = df[['h', 'i']]
    tm.assert_frame_equal(ri, ei)
    with pytest.raises(NotImplementedError, match='^$'):
        df.select_dtypes(include=['period'])