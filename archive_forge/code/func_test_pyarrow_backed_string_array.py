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
def test_pyarrow_backed_string_array(self, pa, string_storage):
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'a': pd.Series(['a', None, 'c'], dtype='string[pyarrow]')})
    with pd.option_context('string_storage', string_storage):
        check_round_trip(df, pa, expected=df.astype(f'string[{string_storage}]'))