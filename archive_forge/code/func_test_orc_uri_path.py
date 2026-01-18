import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
import pandas as pd
from pandas import read_orc
import pandas._testing as tm
from pandas.core.arrays import StringArray
import pyarrow as pa
def test_orc_uri_path():
    expected = pd.DataFrame({'int': list(range(1, 4))})
    with tm.ensure_clean('tmp.orc') as path:
        expected.to_orc(path)
        uri = pathlib.Path(path).as_uri()
        result = read_orc(uri)
    tm.assert_frame_equal(result, expected)