from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import extraroute
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _test_extraroute(self, ipv6=False):
    if ipv6:
        route1 = {'destination': 'ffff:f53b:82e4::56/46', 'nexthop': 'dce7:f53b:82e4::56'}
        route2 = {'destination': 'ffff:f53b:ffff::56/46', 'nexthop': 'dce7:f53b:82e4::56'}
    else:
        route1 = {'destination': '192.168.0.0/24', 'nexthop': '1.1.1.1'}
        route2 = {'destination': '192.168.255.0/24', 'nexthop': '1.1.1.1'}
    self.stub_RouterConstraint_validate()
    self.mockclient.show_router.side_effect = [{'router': {'routes': []}}, {'router': {'routes': [route1.copy()]}}, {'router': {'routes': [route1.copy(), route2.copy()]}}, {'router': {'routes': [route2.copy()]}}]
    self.mockclient.update_router.return_value = None
    t = template_format.parse(neutron_template)
    stack = utils.parse_stack(t)
    if ipv6:
        rsrc1 = self.create_extraroute(t, stack, 'extraroute1', properties={'router_id': '3e46229d-8fce-4733-819a-b5fe630550f8', 'destination': 'ffff:f53b:82e4::56/46', 'nexthop': 'dce7:f53b:82e4::56'})
        self.create_extraroute(t, stack, 'extraroute2', properties={'router_id': '3e46229d-8fce-4733-819a-b5fe630550f8', 'destination': 'ffff:f53b:ffff::56/46', 'nexthop': 'dce7:f53b:82e4::56'})
    else:
        rsrc1 = self.create_extraroute(t, stack, 'extraroute1', properties={'router_id': '3e46229d-8fce-4733-819a-b5fe630550f8', 'destination': '192.168.0.0/24', 'nexthop': '1.1.1.1'})
        self.create_extraroute(t, stack, 'extraroute2', properties={'router_id': '3e46229d-8fce-4733-819a-b5fe630550f8', 'destination': '192.168.255.0/24', 'nexthop': '1.1.1.1'})
    scheduler.TaskRunner(rsrc1.delete)()
    rsrc1.state_set(rsrc1.CREATE, rsrc1.COMPLETE, 'to delete again')
    scheduler.TaskRunner(rsrc1.delete)()
    self.mockclient.show_router.assert_called_with('3e46229d-8fce-4733-819a-b5fe630550f8')
    self.mockclient.update_router.assert_has_calls([mock.call('3e46229d-8fce-4733-819a-b5fe630550f8', {'router': {'routes': [route1.copy()]}}), mock.call('3e46229d-8fce-4733-819a-b5fe630550f8', {'router': {'routes': [route1.copy(), route2.copy()]}}), mock.call('3e46229d-8fce-4733-819a-b5fe630550f8', {'router': {'routes': [route2.copy()]}})])