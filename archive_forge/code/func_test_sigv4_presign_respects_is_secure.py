from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_sigv4_presign_respects_is_secure(self):
    self.config = {'s3': {'use-sigv4': True}}
    conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', host='s3.amazonaws.com', is_secure=True)
    url = conn.generate_url_sigv4(86400, 'GET', bucket='examplebucket', key='test.txt')
    self.assertTrue(url.startswith('https://examplebucket.s3.amazonaws.com/test.txt?'))
    conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', host='s3.amazonaws.com', is_secure=False)
    url = conn.generate_url_sigv4(86400, 'GET', bucket='examplebucket', key='test.txt')
    self.assertTrue(url.startswith('http://examplebucket.s3.amazonaws.com/test.txt?'))