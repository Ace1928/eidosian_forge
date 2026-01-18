from io import BytesIO
import pytest
from pandas import read_csv
@pytest.mark.single_cpu
def test_read_without_creds_from_pub_bucket(s3_public_bucket_with_data, s3so):
    pytest.importorskip('s3fs')
    result = read_csv(f's3://{s3_public_bucket_with_data.name}/tips.csv', nrows=3, storage_options=s3so)
    assert len(result) == 3