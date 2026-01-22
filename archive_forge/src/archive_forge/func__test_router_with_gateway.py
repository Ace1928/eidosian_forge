import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _test_router_with_gateway(self, for_delete=False, for_update=False):
    t = template_format.parse(neutron_external_gateway_template)
    stack = utils.parse_stack(t)

    def find_rsrc(resource, name_or_id, cmd_resource=None):
        id_mapping = {'subnet': 'sub1234', 'network': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}
        return id_mapping.get(resource)
    self.find_rsrc_mock.side_effect = find_rsrc
    base_info = {'router': {'status': 'BUILD', 'external_gateway_info': None, 'name': 'Test Router', 'admin_state_up': True, 'tenant_id': '3e21026f2dc94372b105808c0e721661', 'id': '3e46229d-8fce-4733-819a-b5fe630550f8'}}
    external_gw_info = {'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'enable_snat': True, 'external_fixed_ips': [{'ip_address': '192.168.10.99', 'subnet_id': 'sub1234'}]}
    active_info = copy.deepcopy(base_info)
    active_info['router']['status'] = 'ACTIVE'
    active_info['router']['external_gateway_info'] = external_gw_info
    ex_gw_info1 = copy.deepcopy(external_gw_info)
    ex_gw_info1['network_id'] = '91e47a57-7508-46fe-afc9-fc454e8580e1'
    ex_gw_info1['enable_snat'] = False
    active_info1 = copy.deepcopy(active_info)
    active_info1['router']['external_gateway_info'] = ex_gw_info1
    self.create_mock.return_value = base_info
    if for_delete:
        self.show_mock.side_effect = [active_info, qe.NeutronClientException(status_code=404)]
    elif for_update:
        self.show_mock.side_effect = [active_info, active_info, active_info1]
    else:
        self.show_mock.side_effect = [active_info, active_info]
    return (t, stack)