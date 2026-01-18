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
def test_router_hidden_property_translation(self):
    t = template_format.parse(hidden_property_router_template)
    stack = utils.parse_stack(t)
    rsrc = stack['router']
    self.assertIsNone(rsrc.properties['l3_agent_id'])
    self.assertEqual([u'792ff887-6c85-4a56-b518-23f24fa65581'], rsrc.properties['l3_agent_ids'])