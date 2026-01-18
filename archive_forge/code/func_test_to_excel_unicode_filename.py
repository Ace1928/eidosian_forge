from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
def test_to_excel_unicode_filename(self, ext):
    with tm.ensure_clean('ƒu.' + ext) as filename:
        try:
            with open(filename, 'wb'):
                pass
        except UnicodeEncodeError:
            pytest.skip('No unicode file names on this system')
        df = DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
        df.to_excel(filename, sheet_name='test1', float_format='%.2f')
        with ExcelFile(filename) as reader:
            result = pd.read_excel(reader, sheet_name='test1', index_col=0)
    expected = DataFrame([[0.12, 0.23, 0.57], [12.32, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
    tm.assert_frame_equal(result, expected)