from io import BytesIO
import pytest
from pandas import read_csv
@pytest.mark.single_cpu
def test_read_with_creds_from_pub_bucket(s3_public_bucket_with_data, s3so):
    pytest.importorskip('s3fs')
    df = read_csv(f's3://{s3_public_bucket_with_data.name}/tips.csv', nrows=5, header=None, storage_options=s3so)
    assert len(df) == 5