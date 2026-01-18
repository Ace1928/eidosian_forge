from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', ['a\n1\n"b"a', 'a,b,c\ncat,foo,bar\ndog,foo,"baz'])
@pytest.mark.parametrize('skipfooter', [0, 1])
def test_skipfooter_bad_row(python_parser_only, data, skipfooter):
    parser = python_parser_only
    if skipfooter:
        msg = 'parsing errors in the skipped footer rows'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)
    else:
        msg = 'unexpected end of data|expected after'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)