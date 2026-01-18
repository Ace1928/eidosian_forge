from unittest import mock
from neutronclient.neutron import v2_0 as neutronV20
from osc_lib import exceptions
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os.octavia import OctaviaClientPlugin
from heat.engine.resources.openstack.octavia import loadbalancer
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_update_complete(self):
    self._create_stack()
    prop_diff = {'name': 'lb', 'description': 'a loadbalancer', 'admin_state_up': False}
    self.octavia_client.load_balancer_show.side_effect = [{'provisioning_status': 'ACTIVE'}, {'provisioning_status': 'PENDING_UPDATE'}]
    self.lb.handle_update(None, None, prop_diff)
    self.assertTrue(self.lb.check_update_complete(prop_diff))
    self.assertFalse(self.lb.check_update_complete(prop_diff))
    self.assertTrue(self.lb.check_update_complete({}))