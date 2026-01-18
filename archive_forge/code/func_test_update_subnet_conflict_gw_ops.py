import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_update_subnet_conflict_gw_ops(self):
    self.assertRaises(exceptions.SDKException, self.cloud.update_subnet, self.subnet_id, gateway_ip='192.168.199.3', disable_gateway_ip=True)