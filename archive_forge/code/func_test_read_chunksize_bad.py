from io import StringIO
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('chunksize', [1.3, 'foo', 0])
def test_read_chunksize_bad(all_parsers, chunksize):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    msg = "'chunksize' must be an integer >=1"
    if parser.engine == 'pyarrow':
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), chunksize=chunksize) as _:
            pass