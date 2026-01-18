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
def test_to_csv_string_array_utf8(self):
    str_array = [{'names': ['foo', 'bar']}, {'names': ['baz', 'qux']}]
    df = DataFrame(str_array)
    expected_utf8 = ',names\n0,"[\'foo\', \'bar\']"\n1,"[\'baz\', \'qux\']"\n'
    with tm.ensure_clean('unicode_test.csv') as path:
        df.to_csv(path, encoding='utf-8')
        with open(path, encoding='utf-8') as f:
            assert f.read() == expected_utf8