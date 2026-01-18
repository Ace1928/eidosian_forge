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
@pytest.mark.single_cpu
@pytest.mark.parametrize('partition_col', [['A'], []])
def test_s3_roundtrip_for_dir(self, df_compat, s3_public_bucket, pa, partition_col, s3so):
    pytest.importorskip('s3fs')
    expected_df = df_compat.copy()
    if partition_col:
        expected_df = expected_df.astype(dict.fromkeys(partition_col, np.int32))
        partition_col_type = 'category'
        expected_df[partition_col] = expected_df[partition_col].astype(partition_col_type)
    check_round_trip(df_compat, pa, expected=expected_df, path=f's3://{s3_public_bucket.name}/parquet_dir', read_kwargs={'storage_options': s3so}, write_kwargs={'partition_cols': partition_col, 'compression': None, 'storage_options': s3so}, check_like=True, repeat=1)