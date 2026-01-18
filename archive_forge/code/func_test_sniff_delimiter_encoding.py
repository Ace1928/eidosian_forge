from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('encoding', [None, 'utf-8'])
def test_sniff_delimiter_encoding(python_parser_only, encoding):
    parser = python_parser_only
    data = 'ignore this\nignore this too\nindex|A|B|C\nfoo|1|2|3\nbar|4|5|6\nbaz|7|8|9\n'
    if encoding is not None:
        data = data.encode(encoding)
        data = BytesIO(data)
        data = TextIOWrapper(data, encoding=encoding)
    else:
        data = StringIO(data)
    result = parser.read_csv(data, index_col=0, sep=None, skiprows=2, encoding=encoding)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=Index(['foo', 'bar', 'baz'], name='index'))
    tm.assert_frame_equal(result, expected)