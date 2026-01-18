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
def test_skip_bad_lines(self):
    data = 'a:b:c\nd:e:f\ng:h:i\nj:k:l:m\nl:m:n\no:p:q:r'
    reader = TextReader(StringIO(data), delimiter=':', header=None)
    msg = 'Error tokenizing data\\. C error: Expected 3 fields in line 4, saw 4'
    with pytest.raises(parser.ParserError, match=msg):
        reader.read()
    reader = TextReader(StringIO(data), delimiter=':', header=None, on_bad_lines=2)
    result = reader.read()
    expected = {0: np.array(['a', 'd', 'g', 'l'], dtype=object), 1: np.array(['b', 'e', 'h', 'm'], dtype=object), 2: np.array(['c', 'f', 'i', 'n'], dtype=object)}
    assert_array_dicts_equal(result, expected)
    with tm.assert_produces_warning(ParserWarning, match='Skipping line'):
        reader = TextReader(StringIO(data), delimiter=':', header=None, on_bad_lines=1)
        reader.read()