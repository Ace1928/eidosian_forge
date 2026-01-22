from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
class TestPresigned(MockServiceWithConfigTestCase):
    connection_class = S3Connection

    def test_presign_respect_query_auth(self):
        self.config = {'s3': {'use-sigv4': False}}
        conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', host='s3.amazonaws.com')
        url_enabled = conn.generate_url(86400, 'GET', bucket='examplebucket', key='test.txt', query_auth=True)
        url_disabled = conn.generate_url(86400, 'GET', bucket='examplebucket', key='test.txt', query_auth=False)
        self.assertIn('Signature=', url_enabled)
        self.assertNotIn('Signature=', url_disabled)