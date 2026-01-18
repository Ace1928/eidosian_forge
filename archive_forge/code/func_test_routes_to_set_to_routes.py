import copy
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import extrarouteset
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
from neutronclient.common import exceptions as ncex
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
def test_routes_to_set_to_routes(self):
    routes = [{'destination': '10.0.1.0/24', 'nexthop': '10.0.0.11'}]
    self.assertEqual(routes, extrarouteset._set_to_routes(extrarouteset._routes_to_set(routes)))