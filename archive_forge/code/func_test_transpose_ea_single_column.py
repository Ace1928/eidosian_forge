import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_transpose_ea_single_column(using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
    result = df.T
    assert not np.shares_memory(get_array(df, 'a'), get_array(result, 0))