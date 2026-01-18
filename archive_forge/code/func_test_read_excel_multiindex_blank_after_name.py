from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('sheet_name,idx_lvl2', [('both_name_blank_after_mi_name', [np.nan, 'b', 'a', 'b']), ('both_name_multiple_blanks', [np.nan] * 4)])
def test_read_excel_multiindex_blank_after_name(self, request, engine, read_ext, sheet_name, idx_lvl2):
    xfail_datetimes_with_pyxlsb(engine, request)
    mi_file = 'testmultiindex' + read_ext
    mi = MultiIndex.from_product([['foo', 'bar'], ['a', 'b']], names=['c1', 'c2'])
    unit = get_exp_unit(read_ext, engine)
    expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp('2015-01-04'), True]], columns=mi, index=MultiIndex.from_arrays((['foo', 'foo', 'bar', 'bar'], idx_lvl2), names=['ilvl1', 'ilvl2']))
    expected[mi[2]] = expected[mi[2]].astype(f'M8[{unit}]')
    result = pd.read_excel(mi_file, sheet_name=sheet_name, index_col=[0, 1], header=[0, 1])
    tm.assert_frame_equal(result, expected)