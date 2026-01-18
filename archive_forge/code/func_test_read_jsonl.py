from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_read_jsonl():
    result = read_json(StringIO('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n'), lines=True)
    expected = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)