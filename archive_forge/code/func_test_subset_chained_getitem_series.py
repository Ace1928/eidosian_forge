import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method', [lambda s: s['a':'c']['a':'b'], lambda s: s.iloc[0:3].iloc[0:2], lambda s: s.loc['a':'c'].loc['a':'b'], lambda s: s.loc['a':'c'].iloc[0:3].iloc[0:2].loc['a':'b'].iloc[0:1]], ids=['getitem', 'iloc', 'loc', 'long-chain'])
def test_subset_chained_getitem_series(backend, method, using_copy_on_write, warn_copy_on_write):
    _, _, Series = backend
    s = Series([1, 2, 3], index=['a', 'b', 'c'])
    s_orig = s.copy()
    subset = method(s)
    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0
    if using_copy_on_write:
        tm.assert_series_equal(s, s_orig)
    else:
        assert s.iloc[0] == 0
    subset = s.iloc[0:3].iloc[0:2]
    with tm.assert_cow_warning(warn_copy_on_write):
        s.iloc[0] = 0
    expected = Series([1, 2], index=['a', 'b'])
    if using_copy_on_write:
        tm.assert_series_equal(subset, expected)
    else:
        assert subset.iloc[0] == 0