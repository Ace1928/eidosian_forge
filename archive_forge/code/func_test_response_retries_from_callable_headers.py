from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_response_retries_from_callable_headers(self):
    for code in self.retry_status_codes:
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            mocked_mexe.side_effect = [self.create_response(code, header=self.test_headers), self.success_response]
            response = self.connection.make_request('HEAD', '/', host=self.default_host)
            self.assertEqual(response, self.success_response)
            self.assertEqual(mocked_mexe.call_count, 2)
            self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)