from openstack.network.v2 import network
from openstack.network.v2 import router
from openstack.network.v2 import subnet
from openstack.tests.functional import base
def _verification(self, interface):
    self.assertEqual(interface['subnet_id'], self.SUB_ID)
    self.assertIn('port_id', interface)