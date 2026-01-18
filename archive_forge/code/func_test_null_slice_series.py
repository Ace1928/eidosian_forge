import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method', [lambda s: s[:], lambda s: s.loc[:], lambda s: s.iloc[:]], ids=['getitem', 'loc', 'iloc'])
def test_null_slice_series(backend, method, using_copy_on_write, warn_copy_on_write):
    _, _, Series = backend
    s = Series([1, 2, 3], index=['a', 'b', 'c'])
    s_orig = s.copy()
    s2 = method(s)
    assert s2 is not s
    with tm.assert_cow_warning(warn_copy_on_write):
        s2.iloc[0] = 0
    if using_copy_on_write:
        tm.assert_series_equal(s, s_orig)
    else:
        assert s.iloc[0] == 0