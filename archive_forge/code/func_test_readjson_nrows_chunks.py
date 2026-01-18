from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
@pytest.mark.parametrize('nrows,chunksize', [(2, 2), (4, 2)])
def test_readjson_nrows_chunks(request, nrows, chunksize, engine):
    if engine == 'pyarrow':
        reason = "Pyarrow only supports a file path as an input and line delimited jsonand doesn't support chunksize parameter."
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))
    jsonl = '{"a": 1, "b": 2}\n        {"a": 3, "b": 4}\n        {"a": 5, "b": 6}\n        {"a": 7, "b": 8}'
    if engine != 'pyarrow':
        with read_json(StringIO(jsonl), lines=True, nrows=nrows, chunksize=chunksize, engine=engine) as reader:
            chunked = pd.concat(reader)
    else:
        with read_json(jsonl, lines=True, nrows=nrows, chunksize=chunksize, engine=engine) as reader:
            chunked = pd.concat(reader)
    expected = DataFrame({'a': [1, 3, 5, 7], 'b': [2, 4, 6, 8]}).iloc[:nrows]
    tm.assert_frame_equal(chunked, expected)