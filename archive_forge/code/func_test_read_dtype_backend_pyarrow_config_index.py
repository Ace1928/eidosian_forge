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
def test_read_dtype_backend_pyarrow_config_index(self, pa):
    df = pd.DataFrame({'a': [1, 2]}, index=pd.Index([3, 4], name='test'), dtype='int64[pyarrow]')
    expected = df.copy()
    import pyarrow
    if Version(pyarrow.__version__) > Version('11.0.0'):
        expected.index = expected.index.astype('int64[pyarrow]')
    check_round_trip(df, engine=pa, read_kwargs={'dtype_backend': 'pyarrow'}, expected=expected)