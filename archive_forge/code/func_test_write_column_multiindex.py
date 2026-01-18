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
def test_write_column_multiindex(self, engine):
    mi_columns = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)])
    df = pd.DataFrame(np.random.default_rng(2).standard_normal((4, 3)), columns=mi_columns)
    if engine == 'fastparquet':
        self.check_error_on_write(df, engine, TypeError, 'Column name must be a string')
    elif engine == 'pyarrow':
        check_round_trip(df, engine)