from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_profile_name_gets_passed(self):
    region = SQSRegionInfo(name='us-west-2', endpoint='us-west-2.queue.amazonaws.com')
    self.service_connection = SQSConnection(https_connection_factory=self.https_connection_factory, region=region, profile_name=self.profile_name)
    self.initialize_service_connection()
    self.set_http_response(status_code=200)
    self.assertEquals(self.service_connection.profile_name, self.profile_name)