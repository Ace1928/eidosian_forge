import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
class TestInterfaceAttach(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.latest'

    def test_interface_attach(self):
        server = self._create_server()
        output = self.nova('interface-attach --net-id %s %s' % (self.network.id, server.id))
        for key in ('ip_address', 'mac_addr', 'port_id', 'port_state'):
            self._get_value_from_the_table(output, key)
        self.assertEqual(self.network.id, self._get_value_from_the_table(output, 'net_id'))