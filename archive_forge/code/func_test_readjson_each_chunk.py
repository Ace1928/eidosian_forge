from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_readjson_each_chunk(request, lines_json_df, engine):
    if engine == 'pyarrow':
        reason = "Pyarrow only supports a file path as an input and line delimited jsonand doesn't support chunksize parameter."
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))
    with read_json(StringIO(lines_json_df), lines=True, chunksize=2, engine=engine) as reader:
        chunks = list(reader)
    assert chunks[0].shape == (2, 2)
    assert chunks[1].shape == (1, 2)