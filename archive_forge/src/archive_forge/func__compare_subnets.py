import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def _compare_subnets(self, exp, real):
    self.assertDictEqual(_subnet.Subnet(**exp).to_dict(computed=False), real.to_dict(computed=False))