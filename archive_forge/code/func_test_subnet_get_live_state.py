import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import openstacksdk
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
def test_subnet_get_live_state(self):
    template = '\n        heat_template_version: 2015-04-30\n        resources:\n          net:\n            type: OS::Neutron::Net\n            properties:\n              name: test\n          subnet:\n            type: OS::Neutron::Subnet\n            properties:\n              network_id: { get_resource: net }\n              cidr: 10.0.0.0/25\n              value_specs:\n                test_value_spec: value_spec_value\n        '
    t = template_format.parse(template)
    stack = utils.parse_stack(t)
    rsrc = stack['subnet']
    stack.create()
    subnet_resp = {'subnet': {'name': 'subnet-subnet-la5usdgifhrd', 'enable_dhcp': True, 'network_id': 'dffd43b3-6206-4402-87e6-8a16ddf3bd68', 'tenant_id': '30f466e3d14b4251853899f9c26e2b66', 'dns_nameservers': [], 'ipv6_ra_mode': None, 'allocation_pools': [{'start': '10.0.0.2', 'end': '10.0.0.126'}], 'gateway_ip': '10.0.0.1', 'ipv6_address_mode': None, 'ip_version': 4, 'host_routes': [], 'prefixlen': None, 'cidr': '10.0.0.0/25', 'id': 'b255342b-31b7-4674-8ea4-a144bca658b0', 'subnetpool_id': None, 'test_value_spec': 'value_spec_value'}}
    rsrc.client().show_subnet = mock.MagicMock(return_value=subnet_resp)
    rsrc.resource_id = '1234'
    reality = rsrc.get_live_state(rsrc.properties)
    expected = {'enable_dhcp': True, 'dns_nameservers': [], 'allocation_pools': [{'start': '10.0.0.2', 'end': '10.0.0.126'}], 'gateway_ip': '10.0.0.1', 'host_routes': [], 'value_specs': {'test_value_spec': 'value_spec_value'}}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    for key in expected:
        self.assertEqual(expected[key], reality[key])