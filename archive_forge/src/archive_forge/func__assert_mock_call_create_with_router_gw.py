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
def _assert_mock_call_create_with_router_gw(self):
    self.create_mock.assert_called_with({'router': {'name': 'Test Router', 'external_gateway_info': {'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'enable_snat': True, 'external_fixed_ips': [{'ip_address': '192.168.10.99', 'subnet_id': 'sub1234'}]}, 'admin_state_up': True}})