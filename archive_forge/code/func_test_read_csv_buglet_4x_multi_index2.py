from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_read_csv_buglet_4x_multi_index2(python_parser_only):
    data = '      A B C\na b c\n1 3 7 0 3 6\n3 1 4 1 5 9'
    parser = python_parser_only
    expected = DataFrame.from_records([(1, 3, 7, 0, 3, 6), (3, 1, 4, 1, 5, 9)], columns=list('abcABC'), index=list('abc'))
    result = parser.read_csv(StringIO(data), sep='\\s+')
    tm.assert_frame_equal(result, expected)