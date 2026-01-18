import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('args_kwargs', [[None, {'par': 10}], [(10,), None]])
def test_rolling_apply_args_kwargs(args_kwargs):

    def numpysum(x, par):
        return np.sum(x + par)
    df = DataFrame({'gr': [1, 1], 'a': [1, 2]})
    idx = Index(['gr', 'a'])
    expected = DataFrame([[11.0, 11.0], [11.0, 12.0]], columns=idx)
    result = df.rolling(1).apply(numpysum, args=args_kwargs[0], kwargs=args_kwargs[1])
    tm.assert_frame_equal(result, expected)
    midx = MultiIndex.from_tuples([(1, 0), (1, 1)], names=['gr', None])
    expected = Series([11.0, 12.0], index=midx, name='a')
    gb_rolling = df.groupby('gr')['a'].rolling(1)
    result = gb_rolling.apply(numpysum, args=args_kwargs[0], kwargs=args_kwargs[1])
    tm.assert_series_equal(result, expected)