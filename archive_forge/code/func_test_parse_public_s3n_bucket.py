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
def test_parse_public_s3n_bucket(self, s3_public_bucket_with_data, tips_df, s3so):
    df = read_csv(f's3n://{s3_public_bucket_with_data.name}/tips.csv', nrows=10, storage_options=s3so)
    assert isinstance(df, DataFrame)
    assert not df.empty
    tm.assert_frame_equal(tips_df.iloc[:10], df)