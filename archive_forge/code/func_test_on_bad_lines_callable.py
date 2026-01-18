from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bad_line_func', [lambda x: ['2', '3'], lambda x: x[:2]])
def test_on_bad_lines_callable(python_parser_only, bad_line_func):
    parser = python_parser_only
    data = 'a,b\n1,2\n2,3,4,5,6\n3,4\n'
    bad_sio = StringIO(data)
    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    expected = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
    tm.assert_frame_equal(result, expected)