from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.json._json import JsonReader
@pytest.mark.parametrize('chunksize', [0, -1, 2.2, 'foo'])
def test_readjson_invalid_chunksize(lines_json_df, chunksize, engine):
    msg = "'chunksize' must be an integer >=1"
    with pytest.raises(ValueError, match=msg):
        with read_json(StringIO(lines_json_df), lines=True, chunksize=chunksize, engine=engine) as _:
            pass