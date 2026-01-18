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
def test_to_csv_write_to_open_file_with_newline_py3(self):
    df = DataFrame({'a': ['x', 'y', 'z']})
    expected_rows = ['x', 'y', 'z']
    expected = 'manual header\n' + tm.convert_rows_list_to_csv_str(expected_rows)
    with tm.ensure_clean('test.txt') as path:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('manual header\n')
            df.to_csv(f, header=None, index=None)
        with open(path, 'rb') as f:
            assert f.read() == bytes(expected, 'utf-8')