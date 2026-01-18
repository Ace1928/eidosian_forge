from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='fix when arrow is default')
def test_to_string_index_with_nan(self):
    df = DataFrame({'id1': {0: '1a3', 1: '9h4'}, 'id2': {0: np.nan, 1: 'd67'}, 'id3': {0: '78d', 1: '79d'}, 'value': {0: 123, 1: 64}})
    y = df.set_index(['id1', 'id2', 'id3'])
    result = y.to_string()
    expected = '             value\nid1 id2 id3       \n1a3 NaN 78d    123\n9h4 d67 79d     64'
    assert result == expected
    y = df.set_index('id2')
    result = y.to_string()
    expected = '     id1  id3  value\nid2                 \nNaN  1a3  78d    123\nd67  9h4  79d     64'
    assert result == expected
    y = df.set_index(['id1', 'id2']).set_index('id3', append=True)
    result = y.to_string()
    expected = '             value\nid1 id2 id3       \n1a3 NaN 78d    123\n9h4 d67 79d     64'
    assert result == expected
    df2 = df.copy()
    df2.loc[:, 'id2'] = np.nan
    y = df2.set_index('id2')
    result = y.to_string()
    expected = '     id1  id3  value\nid2                 \nNaN  1a3  78d    123\nNaN  9h4  79d     64'
    assert result == expected
    df2 = df.copy()
    df2.loc[:, 'id2'] = np.nan
    y = df2.set_index(['id2', 'id3'])
    result = y.to_string()
    expected = '         id1  value\nid2 id3            \nNaN 78d  1a3    123\n    79d  9h4     64'
    assert result == expected
    df = DataFrame({'id1': {0: np.nan, 1: '9h4'}, 'id2': {0: np.nan, 1: 'd67'}, 'id3': {0: np.nan, 1: '79d'}, 'value': {0: 123, 1: 64}})
    y = df.set_index(['id1', 'id2', 'id3'])
    result = y.to_string()
    expected = '             value\nid1 id2 id3       \nNaN NaN NaN    123\n9h4 d67 79d     64'
    assert result == expected