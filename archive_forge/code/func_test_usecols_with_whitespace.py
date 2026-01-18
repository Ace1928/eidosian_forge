from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_usecols_with_whitespace(all_parsers):
    parser = all_parsers
    data = 'a  b  c\n4  apple  bat  5.7\n8  orange  cow  10'
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    if parser.engine == 'pyarrow':
        msg = "The 'delim_whitespace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
                parser.read_csv(StringIO(data), delim_whitespace=True, usecols=('a', 'b'))
        return
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), delim_whitespace=True, usecols=('a', 'b'))
    expected = DataFrame({'a': ['apple', 'orange'], 'b': ['bat', 'cow']}, index=[4, 8])
    tm.assert_frame_equal(result, expected)