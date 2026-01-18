import io
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
from pandas.io.excel import ExcelFile
from pandas.io.excel._base import inspect_excel_format
@pytest.mark.parametrize('file_header', [b'\t\x00\x04\x00\x07\x00\x10\x00', b'\t\x02\x06\x00\x00\x00\x10\x00', b'\t\x04\x06\x00\x00\x00\x10\x00', b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'])
def test_read_old_xls_files(file_header):
    f = io.BytesIO(file_header)
    assert inspect_excel_format(f) == 'xls'