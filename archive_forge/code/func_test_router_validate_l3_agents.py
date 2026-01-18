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
def test_router_validate_l3_agents(self):
    t = template_format.parse(neutron_template)
    props = t['resources']['router']['properties']
    props['l3_agent_ids'] = ['id1', 'id2']
    stack = utils.parse_stack(t)
    rsrc = stack['router']
    exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertIn('Non HA routers can only have one L3 agent', str(exc))
    self.assertIsNone(rsrc.properties.get(rsrc.L3_AGENT_ID))