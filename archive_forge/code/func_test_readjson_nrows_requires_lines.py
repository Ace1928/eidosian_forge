from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
def test_readjson_nrows_requires_lines(engine):
    jsonl = '{"a": 1, "b": 2}\n        {"a": 3, "b": 4}\n        {"a": 5, "b": 6}\n        {"a": 7, "b": 8}'
    msg = 'nrows can only be passed if lines=True'
    with pytest.raises(ValueError, match=msg):
        read_json(jsonl, lines=False, nrows=2, engine=engine)