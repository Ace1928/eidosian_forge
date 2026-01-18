import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_remove_router_interface_missing_argument(self):
    self.assertRaises(ValueError, self.cloud.remove_router_interface, {'id': '123'})