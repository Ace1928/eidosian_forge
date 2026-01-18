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
def test_s3_roundtrip(self, df_compat, s3_public_bucket, fp, s3so):
    check_round_trip(df_compat, fp, path=f's3://{s3_public_bucket.name}/fastparquet.parquet', read_kwargs={'storage_options': s3so}, write_kwargs={'compression': None, 'storage_options': s3so})