import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('cols,expected', [('a', Series([1, 1, 1], name='a')), (['a', 'c'], DataFrame({'a': [1, 1, 1], 'c': [1, 1, 1]}))])
@pytest.mark.parametrize('agg_func', ['count', 'rank', 'size'])
def test_transform_numeric_ret(cols, expected, agg_func):
    df = DataFrame({'a': date_range('2018-01-01', periods=3), 'b': range(3), 'c': range(7, 10)})
    result = df.groupby('b')[cols].transform(agg_func)
    if agg_func == 'rank':
        expected = expected.astype('float')
    elif agg_func == 'size' and cols == ['a', 'c']:
        expected = expected['a'].rename(None)
    tm.assert_equal(result, expected)