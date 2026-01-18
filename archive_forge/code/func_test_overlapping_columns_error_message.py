from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_overlapping_columns_error_message(self):
    df = DataFrame({'key': [1, 2, 3], 'v1': [4, 5, 6], 'v2': [7, 8, 9]})
    df2 = DataFrame({'key': [1, 2, 3], 'v1': [4, 5, 6], 'v2': [7, 8, 9]})
    df.columns = ['key', 'foo', 'foo']
    df2.columns = ['key', 'bar', 'bar']
    expected = DataFrame({'key': [1, 2, 3], 'v1': [4, 5, 6], 'v2': [7, 8, 9], 'v3': [4, 5, 6], 'v4': [7, 8, 9]})
    expected.columns = ['key', 'foo', 'foo', 'bar', 'bar']
    tm.assert_frame_equal(merge(df, df2), expected)
    df2.columns = ['key1', 'foo', 'foo']
    msg = "Data columns not unique: Index\\(\\['foo'\\], dtype='object|string'\\)"
    with pytest.raises(MergeError, match=msg):
        merge(df, df2)