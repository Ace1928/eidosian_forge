import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('vc_sort', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
def test_value_counts_sort_categorical(sort, vc_sort, normalize):
    df = DataFrame({'a': [2, 1, 1, 1], 0: [3, 4, 3, 3]}, dtype='category')
    gb = df.groupby('a', sort=sort, observed=True)
    result = gb.value_counts(sort=vc_sort, normalize=normalize)
    if normalize:
        values = [2 / 3, 1 / 3, 1.0, 0.0]
    else:
        values = [2, 1, 1, 0]
    name = 'proportion' if normalize else 'count'
    expected = DataFrame({'a': Categorical([1, 1, 2, 2]), 0: Categorical([3, 4, 3, 4]), name: values}).set_index(['a', 0])[name]
    if sort and vc_sort:
        taker = [0, 1, 2, 3]
    elif sort and (not vc_sort):
        taker = [0, 1, 2, 3]
    elif not sort and vc_sort:
        taker = [0, 2, 1, 3]
    else:
        taker = [2, 3, 0, 1]
    expected = expected.take(taker)
    tm.assert_series_equal(result, expected)