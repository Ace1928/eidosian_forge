import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
def test_warn_bad_lines(all_parsers):
    parser = all_parsers
    data = 'a\n1\n1,2,3\n4\n5,6,7'
    expected = DataFrame({'a': [1, 4]})
    match_msg = 'Skipping line'
    expected_warning = ParserWarning
    if parser.engine == 'pyarrow':
        match_msg = 'Expected 1 columns, but found 3: 1,2,3'
        expected_warning = (ParserWarning, DeprecationWarning)
    with tm.assert_produces_warning(expected_warning, match=match_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), on_bad_lines='warn')
    tm.assert_frame_equal(result, expected)