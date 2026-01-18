from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kwargs', [{'sep': None}, {'delimiter': '|'}])
def test_sniff_delimiter(python_parser_only, kwargs):
    data = 'index|A|B|C\nfoo|1|2|3\nbar|4|5|6\nbaz|7|8|9\n'
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=Index(['foo', 'bar', 'baz'], name='index'))
    tm.assert_frame_equal(result, expected)