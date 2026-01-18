import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('key, data, dropna, expected', [(['x', 'x', 'x'], [Timestamp('2019-01-01'), NaT, Timestamp('2019-01-01')], True, Series([1], index=pd.Index(['x'], name='key'), name='data')), (['x', 'x', 'x'], [dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1)], True, Series([1], index=pd.Index(['x'], name='key'), name='data')), (['x', 'x', 'x', 'y', 'y'], [dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1)], False, Series([2, 2], index=pd.Index(['x', 'y'], name='key'), name='data')), (['x', 'x', 'x', 'x', 'y'], [dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1)], False, Series([2, 1], index=pd.Index(['x', 'y'], name='key'), name='data'))])
def test_nunique_with_NaT(key, data, dropna, expected):
    df = DataFrame({'key': key, 'data': data})
    result = df.groupby(['key'])['data'].nunique(dropna=dropna)
    tm.assert_series_equal(result, expected)