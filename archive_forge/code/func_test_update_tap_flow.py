from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import tap_flow as _tap_flow
from openstack.network.v2 import tap_service as _tap_service
from openstack.tests.functional import base
def test_update_tap_flow(self):
    description = 'My tap flow'
    sot = self.user_cloud.network.update_tap_flow(self.TAP_FLOW.id, description=description)
    self.assertEqual(description, sot.description)