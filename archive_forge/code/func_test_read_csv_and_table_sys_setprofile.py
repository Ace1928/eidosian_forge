from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
@pytest.mark.parametrize('read_func', ['read_csv', 'read_table'])
def test_read_csv_and_table_sys_setprofile(all_parsers, read_func):
    parser = all_parsers
    data = 'a b\n0 1'
    sys.setprofile(lambda *a, **k: None)
    result = getattr(parser, read_func)(StringIO(data))
    sys.setprofile(None)
    expected = DataFrame({'a b': ['0 1']})
    tm.assert_frame_equal(result, expected)