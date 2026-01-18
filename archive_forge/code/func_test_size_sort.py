import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('by', ['A', 'B', ['A', 'B']])
@pytest.mark.parametrize('sort', [True, False])
def test_size_sort(sort, by):
    df = DataFrame(np.random.choice(20, (1000, 3)), columns=list('ABC'))
    left = df.groupby(by=by, sort=sort).size()
    right = df.groupby(by=by, sort=sort)['C'].apply(lambda a: a.shape[0])
    tm.assert_series_equal(left, right, check_names=False)