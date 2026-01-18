import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('window', [2, '2s'])
def test_rolling_apply_with_pandas_objects(window):
    df = DataFrame({'A': np.random.default_rng(2).standard_normal(5), 'B': np.random.default_rng(2).integers(0, 10, size=5)}, index=date_range('20130101', periods=5, freq='s'))

    def f(x):
        if x.index[0] == df.index[0]:
            return np.nan
        return x.iloc[-1]
    result = df.rolling(window).apply(f, raw=False)
    expected = df.iloc[2:].reindex_like(df)
    tm.assert_frame_equal(result, expected)
    with tm.external_error_raised(AttributeError):
        df.rolling(window).apply(f, raw=True)