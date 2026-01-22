from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
class TestHeadBucket(AWSMockServiceTestCase):
    connection_class = S3Connection

    def default_body(self):
        return ''

    def test_head_bucket_success(self):
        self.set_http_response(status_code=200)
        buck = self.service_connection.head_bucket('my-test-bucket')
        self.assertTrue(isinstance(buck, Bucket))
        self.assertEqual(buck.name, 'my-test-bucket')

    def test_head_bucket_forbidden(self):
        self.set_http_response(status_code=403)
        with self.assertRaises(S3ResponseError) as cm:
            self.service_connection.head_bucket('cant-touch-this')
        err = cm.exception
        self.assertEqual(err.status, 403)
        self.assertEqual(err.error_code, 'AccessDenied')
        self.assertEqual(err.message, 'Access Denied')

    def test_head_bucket_notfound(self):
        self.set_http_response(status_code=404)
        with self.assertRaises(S3ResponseError) as cm:
            self.service_connection.head_bucket('totally-doesnt-exist')
        err = cm.exception
        self.assertEqual(err.status, 404)
        self.assertEqual(err.error_code, 'NoSuchBucket')
        self.assertEqual(err.message, 'The specified bucket does not exist')

    def test_head_bucket_other(self):
        self.set_http_response(status_code=405)
        with self.assertRaises(S3ResponseError) as cm:
            self.service_connection.head_bucket('you-broke-it')
        err = cm.exception
        self.assertEqual(err.status, 405)
        self.assertEqual(err.error_code, None)
        self.assertEqual(err.message, '')