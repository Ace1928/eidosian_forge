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
@pytest.mark.parametrize('delim_whitespace', [True, False])
def test_single_char_leading_whitespace(all_parsers, delim_whitespace):
    parser = all_parsers
    data = 'MyColumn\na\nb\na\nb\n'
    expected = DataFrame({'MyColumn': list('abab')})
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    if parser.engine == 'pyarrow':
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
                parser.read_csv(StringIO(data), skipinitialspace=True, delim_whitespace=delim_whitespace)
        return
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), skipinitialspace=True, delim_whitespace=delim_whitespace)
    tm.assert_frame_equal(result, expected)