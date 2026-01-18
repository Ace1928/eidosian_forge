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
@pytest.mark.parametrize('kwargs,expected', [({'header': None, 'delim_whitespace': True, 'skiprows': [0, 1, 2, 3, 5, 6], 'skip_blank_lines': True}, DataFrame([[1.0, 2.0, 4.0], [5.1, np.nan, 10.0]])), ({'delim_whitespace': True, 'skiprows': [1, 2, 3, 5, 6], 'skip_blank_lines': True}, DataFrame({'A': [1.0, 5.1], 'B': [2.0, np.nan], 'C': [4.0, 10]}))])
def test_trailing_spaces(all_parsers, kwargs, expected):
    data = 'A B C  \nrandom line with trailing spaces    \nskip\n1,2,3\n1,2.,4.\nrandom line with trailing tabs\t\t\t\n   \n5.1,NaN,10.0\n'
    parser = all_parsers
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    if parser.engine == 'pyarrow':
        msg = "The 'delim_whitespace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
                parser.read_csv(StringIO(data.replace(',', '  ')), **kwargs)
        return
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data.replace(',', '  ')), **kwargs)
    tm.assert_frame_equal(result, expected)