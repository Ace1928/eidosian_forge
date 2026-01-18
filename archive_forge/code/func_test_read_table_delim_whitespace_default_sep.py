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
def test_read_table_delim_whitespace_default_sep(all_parsers):
    f = StringIO('a  b  c\n1 -2 -3\n4  5   6')
    parser = all_parsers
    depr_msg = "The 'delim_whitespace' keyword in pd.read_table is deprecated"
    if parser.engine == 'pyarrow':
        msg = "The 'delim_whitespace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
                parser.read_table(f, delim_whitespace=True)
        return
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_table(f, delim_whitespace=True)
    expected = DataFrame({'a': [1, 4], 'b': [-2, 5], 'c': [-3, 6]})
    tm.assert_frame_equal(result, expected)