import collections
from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
def test_value_counts_datetime64(index_or_series, unit):
    klass = index_or_series
    df = pd.DataFrame({'person_id': ['xxyyzz', 'xxyyzz', 'xxyyzz', 'xxyyww', 'foofoo', 'foofoo'], 'dt': pd.to_datetime(['2010-01-01', '2010-01-01', '2010-01-01', '2009-01-01', '2008-09-09', '2008-09-09']).as_unit(unit), 'food': ['PIE', 'GUM', 'EGG', 'EGG', 'PIE', 'GUM']})
    s = klass(df['dt'].copy())
    s.name = None
    idx = pd.to_datetime(['2010-01-01 00:00:00', '2008-09-09 00:00:00', '2009-01-01 00:00:00']).as_unit(unit)
    expected_s = Series([3, 2, 1], index=idx, name='count')
    tm.assert_series_equal(s.value_counts(), expected_s)
    expected = array(np.array(['2010-01-01 00:00:00', '2009-01-01 00:00:00', '2008-09-09 00:00:00'], dtype=f'datetime64[{unit}]'))
    result = s.unique()
    if isinstance(s, Index):
        tm.assert_index_equal(result, DatetimeIndex(expected))
    else:
        tm.assert_extension_array_equal(result, expected)
    assert s.nunique() == 3
    s = df['dt'].copy()
    s = klass(list(s.values) + [pd.NaT] * 4)
    if klass is Series:
        s = s.dt.as_unit(unit)
    else:
        s = s.as_unit(unit)
    result = s.value_counts()
    assert result.index.dtype == f'datetime64[{unit}]'
    tm.assert_series_equal(result, expected_s)
    result = s.value_counts(dropna=False)
    expected_s = pd.concat([Series([4], index=DatetimeIndex([pd.NaT]).as_unit(unit), name='count'), expected_s])
    tm.assert_series_equal(result, expected_s)
    assert s.dtype == f'datetime64[{unit}]'
    unique = s.unique()
    assert unique.dtype == f'datetime64[{unit}]'
    if isinstance(s, Index):
        exp_idx = DatetimeIndex(expected.tolist() + [pd.NaT]).as_unit(unit)
        tm.assert_index_equal(unique, exp_idx)
    else:
        tm.assert_extension_array_equal(unique[:3], expected)
        assert pd.isna(unique[3])
    assert s.nunique() == 3
    assert s.nunique(dropna=False) == 4