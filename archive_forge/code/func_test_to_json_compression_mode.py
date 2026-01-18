from io import (
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def test_to_json_compression_mode(compression):
    expected = pd.DataFrame({'A': [1]})
    with BytesIO() as buffer:
        expected.to_json(buffer, compression=compression)