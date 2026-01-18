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
def test_to_csv_escapechar(self):
    df = DataFrame({'col': ['a"a', '"bb"']})
    expected = '"","col"\n"0","a\\"a"\n"1","\\"bb\\""\n'
    with tm.ensure_clean('test.csv') as path:
        df.to_csv(path, quoting=1, doublequote=False, escapechar='\\')
        with open(path, encoding='utf-8') as f:
            assert f.read() == expected
    df = DataFrame({'col': ['a,a', ',bb,']})
    expected = ',col\n0,a\\,a\n1,\\,bb\\,\n'
    with tm.ensure_clean('test.csv') as path:
        df.to_csv(path, quoting=3, escapechar='\\')
        with open(path, encoding='utf-8') as f:
            assert f.read() == expected