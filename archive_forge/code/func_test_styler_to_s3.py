import contextlib
import time
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.excel import ExcelWriter
from pandas.io.formats.excel import ExcelFormatter
@pytest.mark.single_cpu
@td.skip_if_not_us_locale
def test_styler_to_s3(s3_public_bucket, s3so):
    mock_bucket_name, target_file = (s3_public_bucket.name, 'test.xlsx')
    df = DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
    styler = df.style.set_sticky(axis='index')
    styler.to_excel(f's3://{mock_bucket_name}/{target_file}', storage_options=s3so)
    timeout = 5
    while True:
        if target_file in (obj.key for obj in s3_public_bucket.objects.all()):
            break
        time.sleep(0.1)
        timeout -= 0.1
        assert timeout > 0, 'Timed out waiting for file to appear on moto'
        result = read_excel(f's3://{mock_bucket_name}/{target_file}', index_col=0, storage_options=s3so)
        tm.assert_frame_equal(result, df)