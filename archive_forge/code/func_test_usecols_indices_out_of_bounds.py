from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('names', [None, ['a', 'b']])
def test_usecols_indices_out_of_bounds(all_parsers, names):
    parser = all_parsers
    data = '\na,b\n1,2\n    '
    err = ParserError
    msg = 'Defining usecols with out-of-bounds'
    if parser.engine == 'pyarrow':
        err = ValueError
        msg = _msg_pyarrow_requires_names
    with pytest.raises(err, match=msg):
        parser.read_csv(StringIO(data), usecols=[0, 2], names=names, header=0)