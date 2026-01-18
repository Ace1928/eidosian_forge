from io import (
import numpy as np
import pytest
import pandas._libs.parsers as parser
from pandas._libs.parsers import TextReader
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.parsers import (
from pandas.io.parsers.c_parser_wrapper import ensure_dtype_objs
def test_numpy_string_dtype(self):
    data = 'a,1\naa,2\naaa,3\naaaa,4\naaaaa,5'

    def _make_reader(**kwds):
        if 'dtype' in kwds:
            kwds['dtype'] = ensure_dtype_objs(kwds['dtype'])
        return TextReader(StringIO(data), delimiter=',', header=None, **kwds)
    reader = _make_reader(dtype='S5,i4')
    result = reader.read()
    assert result[0].dtype == 'S5'
    ex_values = np.array(['a', 'aa', 'aaa', 'aaaa', 'aaaaa'], dtype='S5')
    assert (result[0] == ex_values).all()
    assert result[1].dtype == 'i4'
    reader = _make_reader(dtype='S4')
    result = reader.read()
    assert result[0].dtype == 'S4'
    ex_values = np.array(['a', 'aa', 'aaa', 'aaaa', 'aaaa'], dtype='S4')
    assert (result[0] == ex_values).all()
    assert result[1].dtype == 'S4'