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
def test_read_datetime_multiindex(self, request, engine, read_ext):
    xfail_datetimes_with_pyxlsb(engine, request)
    f = 'test_datetime_mi' + read_ext
    with pd.ExcelFile(f) as excel:
        actual = pd.read_excel(excel, header=[0, 1], index_col=0, engine=engine)
    unit = get_exp_unit(read_ext, engine)
    dti = pd.DatetimeIndex(['2020-02-29', '2020-03-01'], dtype=f'M8[{unit}]')
    expected_column_index = MultiIndex.from_arrays([dti[:1], dti[1:]], names=[dti[0].to_pydatetime(), dti[1].to_pydatetime()])
    expected = DataFrame([], index=[], columns=expected_column_index)
    tm.assert_frame_equal(expected, actual)