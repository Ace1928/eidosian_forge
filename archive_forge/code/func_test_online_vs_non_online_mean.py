import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.slow
@pytest.mark.parametrize('obj', [DataFrame({'a': range(5), 'b': range(5)}), Series(range(5), name='foo')])
def test_online_vs_non_online_mean(self, obj, nogil, parallel, nopython, adjust, ignore_na):
    expected = obj.ewm(0.5, adjust=adjust, ignore_na=ignore_na).mean()
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    online_ewm = obj.head(2).ewm(0.5, adjust=adjust, ignore_na=ignore_na).online(engine_kwargs=engine_kwargs)
    for _ in range(2):
        result = online_ewm.mean()
        tm.assert_equal(result, expected.head(2))
        result = online_ewm.mean(update=obj.tail(3))
        tm.assert_equal(result, expected.tail(3))
        online_ewm.reset()