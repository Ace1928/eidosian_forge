import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(compat.is_platform_windows(), reason="Especially in Windows, file stream should not be passedto csv writer without newline='' option.(https://docs.python.org/3/library/csv.html#csv.writer)")
def test_to_csv_write_to_open_file(self):
    df = DataFrame({'a': ['x', 'y', 'z']})
    expected = 'manual header\nx\ny\nz\n'
    with tm.ensure_clean('test.txt') as path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('manual header\n')
            df.to_csv(f, header=None, index=None)
        with open(path, encoding='utf-8') as f:
            assert f.read() == expected