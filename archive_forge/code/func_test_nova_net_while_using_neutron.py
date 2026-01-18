from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_network
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_nova_net_while_using_neutron(self):
    t = template_format.parse(stack_template)
    t['resources']['share_network']['properties']['nova_network'] = 'n'
    del t['resources']['share_network']['properties']['neutron_network']
    del t['resources']['share_network']['properties']['neutron_subnet']
    stack = utils.parse_stack(t)
    rsrc_defn = stack.t.resource_definitions(stack)['share_network']
    net = self._create_network('share_network', rsrc_defn, stack)
    net.is_using_neutron = mock.Mock(return_value=True)
    msg = 'With Neutron enabled you need to pass Neutron network and Neutron subnet instead of Nova network'
    self.assertRaisesRegex(exception.StackValidationFailed, msg, net.validate)