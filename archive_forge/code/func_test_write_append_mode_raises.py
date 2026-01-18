import contextlib
import pytest
from pandas.compat import is_platform_windows
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.excel import ExcelWriter
def test_write_append_mode_raises(ext):
    msg = 'Append mode is not supported with xlsxwriter!'
    with tm.ensure_clean(ext) as f:
        with pytest.raises(ValueError, match=msg):
            ExcelWriter(f, engine='xlsxwriter', mode='a')