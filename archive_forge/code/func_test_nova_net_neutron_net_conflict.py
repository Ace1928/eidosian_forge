from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_network
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_nova_net_neutron_net_conflict(self):
    t = template_format.parse(stack_template)
    t['resources']['share_network']['properties']['nova_network'] = 1
    stack = utils.parse_stack(t)
    rsrc_defn = stack.t.resource_definitions(stack)['share_network']
    net = self._create_network('share_network', rsrc_defn, stack)
    msg = 'Cannot define the following properties at the same time: neutron_network, nova_network.'
    self.assertRaisesRegex(exception.ResourcePropertyConflict, msg, net.validate)