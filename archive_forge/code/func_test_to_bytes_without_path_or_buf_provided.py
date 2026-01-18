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
def test_to_bytes_without_path_or_buf_provided(self, pa, df_full):
    buf_bytes = df_full.to_parquet(engine=pa)
    assert isinstance(buf_bytes, bytes)
    buf_stream = BytesIO(buf_bytes)
    res = read_parquet(buf_stream)
    expected = df_full.copy()
    expected.loc[1, 'string_with_nan'] = None
    tm.assert_frame_equal(res, expected)