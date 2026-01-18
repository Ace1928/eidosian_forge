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
@pytest.mark.xfail(reason='GH#39155 s3fs upgrade', strict=False)
def test_write_s3_parquet_fails(self, tips_df, s3so):
    pytest.importorskip('pyarrow')
    import botocore
    error = (FileNotFoundError, botocore.exceptions.ClientError)
    with pytest.raises(error, match='The specified bucket does not exist'):
        tips_df.to_parquet('s3://an_s3_bucket_data_doesnt_exit/not_real.parquet', storage_options=s3so)