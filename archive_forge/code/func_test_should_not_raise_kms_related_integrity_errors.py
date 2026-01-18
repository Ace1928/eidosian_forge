import io
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.compat import StringIO
from boto.exception import BotoServerError
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.key import Key
def test_should_not_raise_kms_related_integrity_errors(self):
    self.set_http_response(status_code=200, header=[('x-amz-server-side-encryption-aws-kms-key-id', 'key'), ('etag', 'not equal to key.md5')])
    bucket = Bucket(self.service_connection, 'mybucket')
    key = bucket.new_key('test_kms')
    file_content = StringIO('Some content to upload.')
    key.send_file(file_content)