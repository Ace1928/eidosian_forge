import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_show_minimal(self):
    uuid = self._create_server().id
    server_output = self.nova('show --minimal %s' % uuid)
    server_output_flavor = self._get_value_from_the_table(server_output, 'flavor')
    self.assertEqual(self.flavor.name, server_output_flavor)