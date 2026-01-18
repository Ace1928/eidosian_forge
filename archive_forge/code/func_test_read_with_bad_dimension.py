import contextlib
from pathlib import Path
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._openpyxl import OpenpyxlReader
@pytest.mark.parametrize('header, expected_data', [(0, {'Title': [np.nan, 'A', 1, 2, 3], 'Unnamed: 1': [np.nan, 'B', 4, 5, 6], 'Unnamed: 2': [np.nan, 'C', 7, 8, 9]}), (2, {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})])
@pytest.mark.parametrize('filename', ['dimension_missing', 'dimension_small', 'dimension_large'])
@pytest.mark.parametrize('read_only', [True, False, None])
def test_read_with_bad_dimension(datapath, ext, header, expected_data, filename, read_only):
    path = datapath('io', 'data', 'excel', f'{filename}{ext}')
    if read_only is None:
        result = pd.read_excel(path, header=header)
    else:
        with contextlib.closing(openpyxl.load_workbook(path, read_only=read_only)) as wb:
            result = pd.read_excel(wb, engine='openpyxl', header=header)
    expected = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected)