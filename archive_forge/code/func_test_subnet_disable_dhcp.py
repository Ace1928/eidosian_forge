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
def test_subnet_disable_dhcp(self):
    t = template_format.parse(neutron_template)
    t['resources']['sub_net']['properties']['enable_dhcp'] = 'False'
    stack = utils.parse_stack(t)
    subnet_info = {'subnet': {'allocation_pools': [{'start': '10.0.3.20', 'end': '10.0.3.150'}], 'host_routes': [{'destination': '10.0.4.0/24', 'nexthop': '10.0.3.20'}], 'cidr': '10.0.3.0/24', 'dns_nameservers': ['8.8.8.8'], 'enable_dhcp': False, 'gateway_ip': '10.0.3.1', 'id': '91e47a57-7508-46fe-afc9-fc454e8580e1', 'ip_version': 4, 'name': 'name', 'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f'}}
    self.create_mock.return_value = subnet_info
    self.show_mock.side_effect = [subnet_info, qe.NeutronClientException(status_code=404)]
    self.patchobject(stack['net'], 'FnGetRefId', return_value='fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    rsrc = self.create_subnet(t, stack, 'sub_net')
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    rsrc.validate()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('91e47a57-7508-46fe-afc9-fc454e8580e1', ref_id)
    self.assertIs(False, rsrc.FnGetAtt('enable_dhcp'))
    scheduler.TaskRunner(rsrc.delete)()