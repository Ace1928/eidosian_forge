import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('dtype', ['int64', 'Int64'])
def test_putmask_no_reference(using_copy_on_write, dtype):
    df = DataFrame({'a': [1, 2], 'b': 1, 'c': 2}, dtype=dtype)
    arr_a = get_array(df, 'a')
    df[df == df] = 5
    if using_copy_on_write:
        assert np.shares_memory(arr_a, get_array(df, 'a'))