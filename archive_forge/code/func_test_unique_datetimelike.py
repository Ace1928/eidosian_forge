from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
def test_unique_datetimelike():
    idx1 = DatetimeIndex(['2015-01-01', '2015-01-01', '2015-01-01', '2015-01-01', 'NaT', 'NaT'])
    idx2 = DatetimeIndex(['2015-01-01', '2015-01-01', '2015-01-02', '2015-01-02', 'NaT', '2015-01-01'], tz='Asia/Tokyo')
    result = MultiIndex.from_arrays([idx1, idx2]).unique()
    eidx1 = DatetimeIndex(['2015-01-01', '2015-01-01', 'NaT', 'NaT'])
    eidx2 = DatetimeIndex(['2015-01-01', '2015-01-02', 'NaT', '2015-01-01'], tz='Asia/Tokyo')
    exp = MultiIndex.from_arrays([eidx1, eidx2])
    tm.assert_index_equal(result, exp)