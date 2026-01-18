from io import (
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
@td.skip_if_not_us_locale
@pytest.mark.single_cpu
def test_with_s3_url(compression, s3_public_bucket, s3so):
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
    with tm.ensure_clean() as path:
        df.to_json(path, compression=compression)
        with open(path, 'rb') as f:
            s3_public_bucket.put_object(Key='test-1', Body=f)
    roundtripped_df = pd.read_json(f's3://{s3_public_bucket.name}/test-1', compression=compression, storage_options=s3so)
    tm.assert_frame_equal(df, roundtripped_df)