from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('match,kwargs', [('the dtype datetime64 is not supported for parsing, pass this column using parse_dates instead', {'dtype': {'A': 'datetime64', 'B': 'float64'}}), ('the dtype datetime64 is not supported for parsing, pass this column using parse_dates instead', {'dtype': {'A': 'datetime64', 'B': 'float64'}, 'parse_dates': ['B']}), ('the dtype timedelta64 is not supported for parsing', {'dtype': {'A': 'timedelta64', 'B': 'float64'}}), (f'the dtype {tm.ENDIAN}U8 is not supported for parsing', {'dtype': {'A': 'U8'}})], ids=['dt64-0', 'dt64-1', 'td64', f'{tm.ENDIAN}U8'])
def test_unsupported_dtype(c_parser_only, match, kwargs):
    parser = c_parser_only
    df = DataFrame(np.random.default_rng(2).random((5, 2)), columns=list('AB'), index=['1A', '1B', '1C', '1D', '1E'])
    with tm.ensure_clean('__unsupported_dtype__.csv') as path:
        df.to_csv(path)
        with pytest.raises(TypeError, match=match):
            parser.read_csv(path, index_col=0, **kwargs)