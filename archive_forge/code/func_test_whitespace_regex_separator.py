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
@pytest.mark.parametrize('data,expected', [('   A   B   C   D\na   1   2   3   4\nb   1   2   3   4\nc   1   2   3   4\n', DataFrame([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], columns=['A', 'B', 'C', 'D'], index=['a', 'b', 'c'])), ('    a b c\n1 2 3 \n4 5  6\n 7 8 9', DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c']))])
def test_whitespace_regex_separator(all_parsers, data, expected):
    parser = all_parsers
    if parser.engine == 'pyarrow':
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep='\\s+')
        return
    result = parser.read_csv(StringIO(data), sep='\\s+')
    tm.assert_frame_equal(result, expected)