from io import BytesIO
import pytest
from pandas import read_csv
def test_streaming_s3_objects():
    pytest.importorskip('botocore', minversion='1.10.47')
    from botocore.response import StreamingBody
    data = [b'foo,bar,baz\n1,2,3\n4,5,6\n', b'just,the,header\n']
    for el in data:
        body = StreamingBody(BytesIO(el), content_length=len(el))
        read_csv(body)