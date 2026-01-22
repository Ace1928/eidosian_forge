from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import instance_action
class InstanceActionExtensionTests(utils.TestCase):

    def setUp(self):
        super(InstanceActionExtensionTests, self).setUp()
        self.cs = fakes.FakeClient(api_versions.APIVersion('2.1'))

    def test_list_instance_actions(self):
        server_uuid = '1234'
        ial = self.cs.instance_action.list(server_uuid)
        self.assert_request_id(ial, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/servers/%s/os-instance-actions' % server_uuid)

    def test_get_instance_action(self):
        server_uuid = '1234'
        request_id = 'req-abcde12345'
        ia = self.cs.instance_action.get(server_uuid, request_id)
        self.assert_request_id(ia, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/servers/%s/os-instance-actions/%s' % (server_uuid, request_id))