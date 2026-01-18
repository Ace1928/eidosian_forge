import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('to_replace', ['a', 100.5])
def test_replace_inplace_reference_no_op(using_copy_on_write, to_replace):
    df = DataFrame({'a': [1.5, 2, 3]})
    arr_a = get_array(df, 'a')
    view = df[:]
    df.replace(to_replace=to_replace, value=15.5, inplace=True)
    assert np.shares_memory(get_array(df, 'a'), arr_a)
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
        assert not view._mgr._has_no_reference(0)