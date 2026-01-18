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
def test_read_parquet_manager(self, pa, using_array_manager):
    df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['A', 'B', 'C'])
    with tm.ensure_clean() as path:
        df.to_parquet(path, engine=pa)
        result = read_parquet(path, pa)
    if using_array_manager:
        assert isinstance(result._mgr, pd.core.internals.ArrayManager)
    else:
        assert isinstance(result._mgr, pd.core.internals.BlockManager)