from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slicing_with_multiindex(self):
    df = DataFrame({'ACCOUNT': ['ACCT1', 'ACCT1', 'ACCT1', 'ACCT2'], 'TICKER': ['ABC', 'MNP', 'XYZ', 'XYZ'], 'val': [1, 2, 3, 4]}, index=date_range('2013-06-19 09:30:00', periods=4, freq='5min'))
    df_multi = df.set_index(['ACCOUNT', 'TICKER'], append=True)
    expected = DataFrame([[1]], index=Index(['ABC'], name='TICKER'), columns=['val'])
    result = df_multi.loc['2013-06-19 09:30:00', 'ACCT1']
    tm.assert_frame_equal(result, expected)
    expected = df_multi.loc[Timestamp('2013-06-19 09:30:00', tz=None), 'ACCT1', 'ABC']
    result = df_multi.loc['2013-06-19 09:30:00', 'ACCT1', 'ABC']
    tm.assert_series_equal(result, expected)
    result = df_multi.loc['2013-06-19', 'ACCT1', 'ABC']
    expected = df_multi.iloc[:1].droplevel([1, 2])
    tm.assert_frame_equal(result, expected)