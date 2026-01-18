import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_show_actions_with_host(self):
    name = self.name_generate()
    server = self.another_nova('boot --flavor %s --image %s --poll %s' % (self.flavor.name, self.image.name, name))
    server_id = self._get_value_from_the_table(server, 'id')
    self.addCleanup(self.client.servers.delete, server_id)
    output = self.nova('instance-action-list %s' % server_id)
    request_id = self._get_column_value_from_single_row_table(output, 'Request_ID')
    output = self.another_nova('instance-action %s %s' % (server_id, request_id))
    self.assertNotIn("'host'", output)
    self.assertIn("'hostId'", output)
    output = self.nova('instance-action %s %s' % (server_id, request_id))
    self.assertIn("'host'", output)
    self.assertIn("'hostId'", output)