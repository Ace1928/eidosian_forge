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
def test_write_index(self, engine, using_copy_on_write, request):
    check_names = engine != 'fastparquet'
    if using_copy_on_write and engine == 'fastparquet':
        request.applymarker(pytest.mark.xfail(reason='fastparquet write into index'))
    df = pd.DataFrame({'A': [1, 2, 3]})
    check_round_trip(df, engine)
    indexes = [[2, 3, 4], pd.date_range('20130101', periods=3), list('abc'), [1, 3, 4]]
    for index in indexes:
        df.index = index
        if isinstance(index, pd.DatetimeIndex):
            df.index = df.index._with_freq(None)
        check_round_trip(df, engine, check_names=check_names)
    df.index = [0, 1, 2]
    df.index.name = 'foo'
    check_round_trip(df, engine)