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
def test_storage_class(self):
    self.set_http_response(status_code=200)
    b = Bucket(self.service_connection, 'mybucket')
    k = b.get_key('fookey')
    k.bucket = mock.MagicMock()
    k.set_contents_from_string('test')
    k.bucket.list.assert_not_called()
    sc_value = k.storage_class
    self.assertEqual(sc_value, 'STANDARD')
    k.bucket.list.assert_called_with(k.name.encode('utf-8'))
    k.bucket.list.reset_mock()
    k.storage_class = 'GLACIER'
    k.set_contents_from_string('test')
    k.bucket.list.assert_not_called()