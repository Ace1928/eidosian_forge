from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.single_cpu
def test_read_from_s3_object(self, read_ext, s3_public_bucket, s3so):
    with open('test1' + read_ext, 'rb') as f:
        s3_public_bucket.put_object(Key='test1' + read_ext, Body=f)
    import s3fs
    s3 = s3fs.S3FileSystem(**s3so)
    with s3.open(f's3://{s3_public_bucket.name}/test1' + read_ext) as f:
        url_table = pd.read_excel(f)
    local_table = pd.read_excel('test1' + read_ext)
    tm.assert_frame_equal(url_table, local_table)