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
def test_parse_public_s3_bucket_nrows(self, s3_public_bucket_with_data, tips_df, s3so):
    for ext, comp in [('', None), ('.gz', 'gzip'), ('.bz2', 'bz2')]:
        df = read_csv(f's3://{s3_public_bucket_with_data.name}/tips.csv' + ext, nrows=10, compression=comp, storage_options=s3so)
        assert isinstance(df, DataFrame)
        assert not df.empty
        tm.assert_frame_equal(tips_df.iloc[:10], df)