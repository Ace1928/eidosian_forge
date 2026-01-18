from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_readjson_chunks_series(request, engine):
    if engine == 'pyarrow':
        reason = "Pyarrow only supports a file path as an input and line delimited jsonand doesn't support chunksize parameter."
        request.applymarker(pytest.mark.xfail(reason=reason))
    s = pd.Series({'A': 1, 'B': 2})
    strio = StringIO(s.to_json(lines=True, orient='records'))
    unchunked = read_json(strio, lines=True, typ='Series', engine=engine)
    strio = StringIO(s.to_json(lines=True, orient='records'))
    with read_json(strio, lines=True, typ='Series', chunksize=1, engine=engine) as reader:
        chunked = pd.concat(reader)
    tm.assert_series_equal(chunked, unchunked)