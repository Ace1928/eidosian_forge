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
def test_partition_on_supported(self, tmp_path, fp, df_full):
    partition_cols = ['bool', 'int']
    df = df_full
    df.to_parquet(tmp_path, engine='fastparquet', compression=None, partition_on=partition_cols)
    assert os.path.exists(tmp_path)
    import fastparquet
    actual_partition_cols = fastparquet.ParquetFile(str(tmp_path), False).cats
    assert len(actual_partition_cols) == 2