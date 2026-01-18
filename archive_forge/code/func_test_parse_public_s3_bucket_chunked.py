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
def test_parse_public_s3_bucket_chunked(self, s3_public_bucket_with_data, tips_df, s3so):
    chunksize = 5
    for ext, comp in [('', None), ('.gz', 'gzip'), ('.bz2', 'bz2')]:
        with read_csv(f's3://{s3_public_bucket_with_data.name}/tips.csv' + ext, chunksize=chunksize, compression=comp, storage_options=s3so) as df_reader:
            assert df_reader.chunksize == chunksize
            for i_chunk in [0, 1, 2]:
                df = df_reader.get_chunk()
                assert isinstance(df, DataFrame)
                assert not df.empty
                true_df = tips_df.iloc[chunksize * i_chunk:chunksize * (i_chunk + 1)]
                tm.assert_frame_equal(true_df, df)