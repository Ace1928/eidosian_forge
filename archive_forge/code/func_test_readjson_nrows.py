from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
@pytest.mark.parametrize('nrows', [1, 2])
def test_readjson_nrows(nrows, engine):
    jsonl = '{"a": 1, "b": 2}\n        {"a": 3, "b": 4}\n        {"a": 5, "b": 6}\n        {"a": 7, "b": 8}'
    result = read_json(StringIO(jsonl), lines=True, nrows=nrows)
    expected = DataFrame({'a': [1, 3, 5, 7], 'b': [2, 4, 6, 8]}).iloc[:nrows]
    tm.assert_frame_equal(result, expected)