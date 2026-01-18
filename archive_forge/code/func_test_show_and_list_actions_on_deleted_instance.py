import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_show_and_list_actions_on_deleted_instance(self):
    server = self._create_server(add_cleanup=False)
    server.delete()
    self.wait_for_resource_delete(server, self.client.servers)
    output = self.nova('instance-action-list %s' % server.id)
    request_id = self._get_column_value_from_single_row_table(output, 'Request_ID')
    output = self.nova('instance-action %s %s' % (server.id, request_id))
    self.assertEqual('create', self._get_value_from_the_table(output, 'action'))