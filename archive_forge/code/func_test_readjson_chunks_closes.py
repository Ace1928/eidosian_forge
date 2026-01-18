from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
@pytest.mark.parametrize('chunksize', [None, 1])
def test_readjson_chunks_closes(chunksize):
    with tm.ensure_clean('test.json') as path:
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df.to_json(path, lines=True, orient='records')
        reader = JsonReader(path, orient=None, typ='frame', dtype=True, convert_axes=True, convert_dates=True, keep_default_dates=True, precise_float=False, date_unit=None, encoding=None, lines=True, chunksize=chunksize, compression=None, nrows=None)
        with reader:
            reader.read()
        assert reader.handles.handle.closed, f"didn't close stream with chunksize = {chunksize}"