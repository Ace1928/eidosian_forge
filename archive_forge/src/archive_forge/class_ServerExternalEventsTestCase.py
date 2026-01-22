from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
class ServerExternalEventsTestCase(utils.TestCase):

    def setUp(self):
        super(ServerExternalEventsTestCase, self).setUp()
        self.cs = fakes.FakeClient(api_versions.APIVersion('2.1'))

    def test_external_event(self):
        events = [{'server_uuid': 'fake-uuid1', 'name': 'test-event', 'status': 'completed', 'tag': 'tag'}, {'server_uuid': 'fake-uuid2', 'name': 'test-event', 'status': 'completed', 'tag': 'tag'}]
        result = self.cs.server_external_events.create(events)
        self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
        self.assertEqual(events, result)
        self.cs.assert_called('POST', '/os-server-external-events')