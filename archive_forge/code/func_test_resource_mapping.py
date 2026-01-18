from unittest import mock
from heat.engine.resources.openstack.neutron.taas import tap_flow
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_mapping(self):
    mapping = tap_flow.resource_mapping()
    self.assertEqual(tap_flow.TapFlow, mapping['OS::Neutron::TaaS::TapFlow'])