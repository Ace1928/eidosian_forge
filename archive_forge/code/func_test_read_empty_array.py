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
@pytest.mark.parametrize('dtype', ['Int64', 'UInt8', 'boolean', 'object', 'datetime64[ns, UTC]', 'float', 'period[D]', 'Float64', 'string'])
def test_read_empty_array(self, pa, dtype):
    df = pd.DataFrame({'value': pd.array([], dtype=dtype)})
    expected = None
    if dtype == 'float':
        expected = pd.DataFrame({'value': pd.array([], dtype='Float64')})
    check_round_trip(df, pa, read_kwargs={'dtype_backend': 'numpy_nullable'}, expected=expected)