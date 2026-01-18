from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_on_bad_lines_index_col_inferred(python_parser_only):
    parser = python_parser_only
    data = 'a,b\n1,2,3\n4,5,6\n'
    bad_sio = StringIO(data)
    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: ['99', '99'])
    expected = DataFrame({'a': [2, 5], 'b': [3, 6]}, index=[1, 4])
    tm.assert_frame_equal(result, expected)