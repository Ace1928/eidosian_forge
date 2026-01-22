from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import instance_action
class InstanceActionExtensionV266Tests(InstanceActionExtensionV258Tests):

    def setUp(self):
        super(InstanceActionExtensionV266Tests, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.66')

    def test_list_instance_actions_with_changes_before(self):
        server_uuid = '1234'
        ias = self.cs.instance_action.list(server_uuid, marker=None, limit=None, changes_since=None, changes_before='2016-02-29T06:23:22')
        self.assert_request_id(ias, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/servers/%s/os-instance-actions?changes-before=%s' % (server_uuid, '2016-02-29T06%3A23%3A22'))
        for ia in ias:
            self.assertIsInstance(ia, instance_action.InstanceAction)