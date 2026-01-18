from io import BytesIO
import logging
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv
@pytest.mark.single_cpu
def test_read_csv_handles_boto_s3_object(self, s3_public_bucket_with_data, tips_file):
    s3_object = s3_public_bucket_with_data.Object('tips.csv')
    with BytesIO(s3_object.get()['Body'].read()) as buffer:
        result = read_csv(buffer, encoding='utf8')
    assert isinstance(result, DataFrame)
    assert not result.empty
    expected = read_csv(tips_file)
    tm.assert_frame_equal(result, expected)