import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('copy', [True, None, False])
def test_transpose_copy_keyword(using_copy_on_write, copy, using_array_manager):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = df.transpose(copy=copy)
    share_memory = using_copy_on_write or copy is False or copy is None
    share_memory = share_memory and (not using_array_manager)
    if share_memory:
        assert np.shares_memory(get_array(df, 'a'), get_array(result, 0))
    else:
        assert not np.shares_memory(get_array(df, 'a'), get_array(result, 0))