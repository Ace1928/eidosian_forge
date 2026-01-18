from contextlib import nullcontext
from datetime import (
import locale
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_representation_to_series(self):
    idx1 = PeriodIndex([], freq='D')
    idx2 = PeriodIndex(['2011-01-01'], freq='D')
    idx3 = PeriodIndex(['2011-01-01', '2011-01-02'], freq='D')
    idx4 = PeriodIndex(['2011-01-01', '2011-01-02', '2011-01-03'], freq='D')
    idx5 = PeriodIndex(['2011', '2012', '2013'], freq='Y')
    idx6 = PeriodIndex(['2011-01-01 09:00', '2012-02-01 10:00', 'NaT'], freq='h')
    idx7 = pd.period_range('2013Q1', periods=1, freq='Q')
    idx8 = pd.period_range('2013Q1', periods=2, freq='Q')
    idx9 = pd.period_range('2013Q1', periods=3, freq='Q')
    exp1 = 'Series([], dtype: period[D])'
    exp2 = '0    2011-01-01\ndtype: period[D]'
    exp3 = '0    2011-01-01\n1    2011-01-02\ndtype: period[D]'
    exp4 = '0    2011-01-01\n1    2011-01-02\n2    2011-01-03\ndtype: period[D]'
    exp5 = '0    2011\n1    2012\n2    2013\ndtype: period[Y-DEC]'
    exp6 = '0    2011-01-01 09:00\n1    2012-02-01 10:00\n2                 NaT\ndtype: period[h]'
    exp7 = '0    2013Q1\ndtype: period[Q-DEC]'
    exp8 = '0    2013Q1\n1    2013Q2\ndtype: period[Q-DEC]'
    exp9 = '0    2013Q1\n1    2013Q2\n2    2013Q3\ndtype: period[Q-DEC]'
    for idx, expected in zip([idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9], [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9]):
        result = repr(Series(idx))
        assert result == expected