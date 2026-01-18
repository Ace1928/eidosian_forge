import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nth_bdays(unit):
    business_dates = pd.date_range(start='4/1/2014', end='6/30/2014', freq='B', unit=unit)
    df = DataFrame(1, index=business_dates, columns=['a', 'b'])
    key = [df.index.year, df.index.month]
    result = df.groupby(key, as_index=False).nth([0, 3, -2, -1])
    expected_dates = pd.to_datetime(['2014/4/1', '2014/4/4', '2014/4/29', '2014/4/30', '2014/5/1', '2014/5/6', '2014/5/29', '2014/5/30', '2014/6/2', '2014/6/5', '2014/6/27', '2014/6/30']).as_unit(unit)
    expected = DataFrame(1, columns=['a', 'b'], index=expected_dates)
    tm.assert_frame_equal(result, expected)