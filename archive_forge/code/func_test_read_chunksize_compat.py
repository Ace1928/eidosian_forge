from io import StringIO
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kwargs', [{}, {'index_col': 0}])
def test_read_chunksize_compat(all_parsers, kwargs):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    if parser.engine == 'pyarrow':
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
                concat(reader)
        return
    with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
        via_reader = concat(reader)
    tm.assert_frame_equal(via_reader, result)