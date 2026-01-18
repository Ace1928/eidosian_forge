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
def test_read_multiindex_header_no_index_names(datapath, ext):
    path = datapath('io', 'data', 'excel', f'multiindex_no_index_names{ext}')
    result = pd.read_excel(path, index_col=[0, 1, 2], header=[0, 1, 2])
    expected = DataFrame([[np.nan, 'x', 'x', 'x'], ['x', np.nan, np.nan, np.nan]], columns=pd.MultiIndex.from_tuples([('X', 'Y', 'A1'), ('X', 'Y', 'A2'), ('XX', 'YY', 'B1'), ('XX', 'YY', 'B2')]), index=pd.MultiIndex.from_tuples([('A', 'AA', 'AAA'), ('A', 'BB', 'BBB')]))
    tm.assert_frame_equal(result, expected)