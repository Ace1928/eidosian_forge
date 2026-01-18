import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_timestamp_groupby_quantile():
    df = DataFrame({'timestamp': pd.date_range(start='2020-04-19 00:00:00', freq='1T', periods=100, tz='UTC').floor('1H'), 'category': list(range(1, 101)), 'value': list(range(101, 201))})
    result = df.groupby('timestamp').quantile([0.2, 0.8])
    expected = DataFrame([{'category': 12.8, 'value': 112.8}, {'category': 48.2, 'value': 148.2}, {'category': 68.8, 'value': 168.8}, {'category': 92.2, 'value': 192.2}], index=pd.MultiIndex.from_tuples([(pd.Timestamp('2020-04-19 00:00:00+00:00'), 0.2), (pd.Timestamp('2020-04-19 00:00:00+00:00'), 0.8), (pd.Timestamp('2020-04-19 01:00:00+00:00'), 0.2), (pd.Timestamp('2020-04-19 01:00:00+00:00'), 0.8)], names=('timestamp', None)))
    tm.assert_frame_equal(result, expected)