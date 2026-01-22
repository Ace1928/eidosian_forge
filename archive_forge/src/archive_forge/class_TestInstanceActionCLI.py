import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestInstanceActionCLI(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.21'

    def _test_cmd_with_not_existing_instance(self, cmd, args):
        try:
            self.nova('%s %s' % (cmd, args))
        except exceptions.CommandFailed as e:
            self.assertIn('ERROR (NotFound):', str(e))
        else:
            self.fail('%s is not failed on non existing instance.' % cmd)

    def test_show_action_with_not_existing_instance(self):
        name_or_uuid = uuidutils.generate_uuid()
        request_id = uuidutils.generate_uuid()
        self._test_cmd_with_not_existing_instance('instance-action', '%s %s' % (name_or_uuid, request_id))

    def test_list_actions_with_not_existing_instance(self):
        name_or_uuid = uuidutils.generate_uuid()
        self._test_cmd_with_not_existing_instance('instance-action-list', name_or_uuid)

    def test_show_and_list_actions_on_deleted_instance(self):
        server = self._create_server(add_cleanup=False)
        server.delete()
        self.wait_for_resource_delete(server, self.client.servers)
        output = self.nova('instance-action-list %s' % server.id)
        request_id = self._get_column_value_from_single_row_table(output, 'Request_ID')
        output = self.nova('instance-action %s %s' % (server.id, request_id))
        self.assertEqual('create', self._get_value_from_the_table(output, 'action'))