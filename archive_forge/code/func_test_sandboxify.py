from boto.mws.connection import MWSConnection, api_call_map, destructure_object
from boto.mws.response import (ResponseElement, GetFeedSubmissionListResult,
from boto.exception import BotoServerError
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from mock import MagicMock
def test_sandboxify(self):
    conn = MWSConnection(https_connection_factory=self.https_connection_factory, aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key', sandbox=True)
    self.assertEqual(conn._sandboxify('a/bogus/path'), 'a/bogus_Sandbox/path')