from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_timestamp_multiindex_indexer():
    idx = MultiIndex.from_product([date_range('2019-01-01T00:15:33', periods=100, freq='h', name='date'), ['x'], [3]])
    df = DataFrame({'foo': np.arange(len(idx))}, idx)
    result = df.loc[pd.IndexSlice['2019-1-2':, 'x', :], 'foo']
    qidx = MultiIndex.from_product([date_range(start='2019-01-02T00:15:33', end='2019-01-05T03:15:33', freq='h', name='date'), ['x'], [3]])
    should_be = pd.Series(data=np.arange(24, len(qidx) + 24), index=qidx, name='foo')
    tm.assert_series_equal(result, should_be)