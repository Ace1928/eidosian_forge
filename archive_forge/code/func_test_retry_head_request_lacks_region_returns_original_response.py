from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_retry_head_request_lacks_region_returns_original_response(self):
    for code in self.retry_status_codes:
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            error_response = self.create_response(code)
            mocked_mexe.side_effect = [error_response, self.create_response(200, header=[])]
            response = self.connection.make_request('HEAD', '/', host=self.default_host)
            self.assertEqual(response, error_response)
            self.assertEqual(mocked_mexe.call_count, 2)
            self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_host)