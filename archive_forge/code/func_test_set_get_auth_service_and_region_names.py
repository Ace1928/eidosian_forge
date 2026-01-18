from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_set_get_auth_service_and_region_names(self):
    self.service_connection.auth_service_name = 'service_name'
    self.service_connection.auth_region_name = 'region_name'
    self.assertEqual(self.service_connection.auth_service_name, 'service_name')
    self.assertEqual(self.service_connection.auth_region_name, 'region_name')