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
def test_read_dtype_backend_pyarrow_config(self, pa, df_full):
    import pyarrow
    df = df_full
    dti = pd.date_range('20130101', periods=3, tz='Europe/Brussels')
    dti = dti._with_freq(None)
    df['datetime_tz'] = dti
    df['bool_with_none'] = [True, None, True]
    pa_table = pyarrow.Table.from_pandas(df)
    expected = pa_table.to_pandas(types_mapper=pd.ArrowDtype)
    if pa_version_under13p0:
        expected['datetime'] = expected['datetime'].astype('timestamp[us][pyarrow]')
        expected['datetime_with_nat'] = expected['datetime_with_nat'].astype('timestamp[us][pyarrow]')
        expected['datetime_tz'] = expected['datetime_tz'].astype(pd.ArrowDtype(pyarrow.timestamp(unit='us', tz='Europe/Brussels')))
    check_round_trip(df, engine=pa, read_kwargs={'dtype_backend': 'pyarrow'}, expected=expected)