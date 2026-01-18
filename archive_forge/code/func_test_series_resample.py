from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize(['obj', 'method', 'npartitions', 'freq', 'closed', 'label'], list(product(['series', 'frame'], ['count', 'mean', 'ohlc'], [2, 5], ['30min', 'h', 'D', 'W', ME], ['right', 'left'], ['right', 'left'])))
def test_series_resample(obj, method, npartitions, freq, closed, label):
    index = pd.date_range('1-1-2000', '2-15-2000', freq='h')
    index = index.union(pd.date_range('4-15-2000', '5-15-2000', freq='h'))
    if obj == 'series':
        ps = pd.Series(range(len(index)), index=index)
    elif obj == 'frame':
        ps = pd.DataFrame({'a': range(len(index))}, index=index)
    ds = dd.from_pandas(ps, npartitions=npartitions)
    result = resample(ds, freq, how=method, closed=closed, label=label)
    expected = resample(ps, freq, how=method, closed=closed, label=label)
    assert_eq(result, expected, check_dtype=False)
    divisions = result.divisions
    assert expected.index[0] == divisions[0]
    assert expected.index[-1] == divisions[-1]