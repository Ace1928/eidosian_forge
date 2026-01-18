from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_retry_changes_host_with_multiple_s3_occurrences(self):
    with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
        mocked_mexe.side_effect = [self.create_response(400, header=self.test_headers), self.success_response]
        response = self.connection.make_request('HEAD', '/', host='a.s3.a.s3.amazonaws.com')
        self.assertEqual(response, self.success_response)
        self.assertEqual(mocked_mexe.call_count, 2)
        self.assertEqual(mocked_mexe.call_args[0][0].host, 'a.s3.a.s3.us-east-2.amazonaws.com')