from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_to_json_append_output_different_columns_reordered():
    df1 = DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    df2 = DataFrame({'col1': [3, 4], 'col2': ['c', 'd']})
    df3 = DataFrame({'col2': ['e', 'f'], 'col3': ['!', '#']})
    df4 = DataFrame({'col4': [True, False]})
    expected = DataFrame({'col4': [True, False, None, None, None, None, None, None], 'col2': [np.nan, np.nan, 'e', 'f', 'c', 'd', 'a', 'b'], 'col3': [np.nan, np.nan, '!', '#', np.nan, np.nan, np.nan, np.nan], 'col1': [None, None, None, None, 3, 4, 1, 2]}).astype({'col4': 'float'})
    with tm.ensure_clean('test.json') as path:
        df4.to_json(path, mode='a', lines=True, orient='records')
        df3.to_json(path, mode='a', lines=True, orient='records')
        df2.to_json(path, mode='a', lines=True, orient='records')
        df1.to_json(path, mode='a', lines=True, orient='records')
        result = read_json(path, lines=True)
        tm.assert_frame_equal(result, expected)