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
def test_s3_roundtrip_explicit_fs(self, df_compat, s3_public_bucket, pa, s3so):
    s3fs = pytest.importorskip('s3fs')
    s3 = s3fs.S3FileSystem(**s3so)
    kw = {'filesystem': s3}
    check_round_trip(df_compat, pa, path=f'{s3_public_bucket.name}/pyarrow.parquet', read_kwargs=kw, write_kwargs=kw)