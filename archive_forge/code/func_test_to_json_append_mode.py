from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
@pytest.mark.parametrize('mode_', ['r', 'x'])
def test_to_json_append_mode(mode_):
    df = DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    msg = f"mode={mode_} is not a valid option.Only 'w' and 'a' are currently supported."
    with pytest.raises(ValueError, match=msg):
        df.to_json(mode=mode_, lines=False, orient='records')