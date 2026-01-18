from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('usecols', [['A', 'C'], [0, 2]])
def test_usecols_subset_names_mismatch_orig_columns(all_parsers, usecols, request):
    data = 'a,b,c,d\n1,2,3,4\n5,6,7,8'
    names = ['A', 'B', 'C', 'D']
    parser = all_parsers
    if parser.engine == 'pyarrow':
        if isinstance(usecols[0], int):
            with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
                parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
            return
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    result = parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
    expected = DataFrame({'A': [1, 5], 'C': [3, 7]})
    tm.assert_frame_equal(result, expected)