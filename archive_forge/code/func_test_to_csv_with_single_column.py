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
def test_to_csv_with_single_column(self):
    df1 = DataFrame([None, 1])
    expected1 = '""\n1.0\n'
    with tm.ensure_clean('test.csv') as path:
        df1.to_csv(path, header=None, index=None)
        with open(path, encoding='utf-8') as f:
            assert f.read() == expected1
    df2 = DataFrame([1, None])
    expected2 = '1.0\n""\n'
    with tm.ensure_clean('test.csv') as path:
        df2.to_csv(path, header=None, index=None)
        with open(path, encoding='utf-8') as f:
            assert f.read() == expected2