from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_sigv4_presign_optional_params(self):
    self.config = {'s3': {'use-sigv4': True}}
    conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', security_token='token', host='s3.amazonaws.com')
    url = conn.generate_url_sigv4(86400, 'GET', bucket='examplebucket', key='test.txt', version_id=2)
    self.assertIn('VersionId=2', url)
    self.assertIn('X-Amz-Security-Token=token', url)