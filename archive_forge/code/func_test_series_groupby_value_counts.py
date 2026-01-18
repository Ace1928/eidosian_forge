from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.slow
@pytest.mark.parametrize('df, keys, bins, n, m', binned, ids=ids)
@pytest.mark.parametrize('isort', [True, False])
@pytest.mark.parametrize('normalize, name', [(True, 'proportion'), (False, 'count')])
@pytest.mark.parametrize('sort', [True, False])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('dropna', [True, False])
def test_series_groupby_value_counts(df, keys, bins, n, m, isort, normalize, name, sort, ascending, dropna):

    def rebuild_index(df):
        arr = list(map(df.index.get_level_values, range(df.index.nlevels)))
        df.index = MultiIndex.from_arrays(arr, names=df.index.names)
        return df
    kwargs = {'normalize': normalize, 'sort': sort, 'ascending': ascending, 'dropna': dropna, 'bins': bins}
    gr = df.groupby(keys, sort=isort)
    left = gr['3rd'].value_counts(**kwargs)
    gr = df.groupby(keys, sort=isort)
    right = gr['3rd'].apply(Series.value_counts, **kwargs)
    right.index.names = right.index.names[:-1] + ['3rd']
    right = right.rename(name)
    left, right = map(rebuild_index, (left, right))
    tm.assert_series_equal(left.sort_index(), right.sort_index())