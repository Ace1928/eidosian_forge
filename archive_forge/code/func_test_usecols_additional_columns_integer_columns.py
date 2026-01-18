from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_usecols_additional_columns_integer_columns(all_parsers):
    parser = all_parsers
    usecols = lambda header: header.strip() in ['0', '1']
    if parser.engine == 'pyarrow':
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO('0,1\nx,y,z'), index_col=False, usecols=usecols)
        return
    result = parser.read_csv(StringIO('0,1\nx,y,z'), index_col=False, usecols=usecols)
    expected = DataFrame({'0': ['x'], '1': 'y'})
    tm.assert_frame_equal(result, expected)