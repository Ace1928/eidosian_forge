from openstack.compute.v2 import server
from openstack.tests.functional.compute import base as ft_base
from openstack.tests.functional.network.v2 import test_network
def test_server_remote_console(self):
    console = self.conn.compute.create_server_remote_console(self.server, protocol='vnc', type='novnc')
    self.assertEqual('vnc', console.protocol)
    self.assertEqual('novnc', console.type)
    self.assertTrue(console.url.startswith('http'))