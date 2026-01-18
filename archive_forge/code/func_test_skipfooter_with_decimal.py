from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('add_footer', [True, False])
def test_skipfooter_with_decimal(python_parser_only, add_footer):
    data = '1#2\n3#4'
    parser = python_parser_only
    expected = DataFrame({'a': [1.2, 3.4]})
    if add_footer:
        kwargs = {'skipfooter': 1}
        data += '\nFooter'
    else:
        kwargs = {}
    result = parser.read_csv(StringIO(data), names=['a'], decimal='#', **kwargs)
    tm.assert_frame_equal(result, expected)