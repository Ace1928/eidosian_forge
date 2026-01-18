import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
from pandas._config import using_copy_on_write
from pandas._config.config import _get_option
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
@pytest.mark.skipif(pa_version_under11p0, reason='not supported before 11.0')
def test_roundtrip_decimal(self, tmp_path, pa):
    import pyarrow as pa
    path = tmp_path / 'decimal.p'
    df = pd.DataFrame({'a': [Decimal('123.00')]}, dtype='string[pyarrow]')
    df.to_parquet(path, schema=pa.schema([('a', pa.decimal128(5))]))
    result = read_parquet(path)
    expected = pd.DataFrame({'a': ['123']}, dtype='string[python]')
    tm.assert_frame_equal(result, expected)