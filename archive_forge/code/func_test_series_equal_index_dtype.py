import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('s1,s2,msg', [(Series(['l1', 'l2'], index=[1, 2]), Series(['l1', 'l2'], index=[1.0, 2.0]), 'Series\\.index are different'), (DataFrame.from_records({'a': [1, 2], 'b': [2.1, 1.5], 'c': ['l1', 'l2']}, index=['a', 'b']).c, DataFrame.from_records({'a': [1.0, 2.0], 'b': [2.1, 1.5], 'c': ['l1', 'l2']}, index=['a', 'b']).c, 'MultiIndex level \\[0\\] are different')])
def test_series_equal_index_dtype(s1, s2, msg, check_index_type):
    kwargs = {'check_index_type': check_index_type}
    if check_index_type:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, **kwargs)
    else:
        tm.assert_series_equal(s1, s2, **kwargs)