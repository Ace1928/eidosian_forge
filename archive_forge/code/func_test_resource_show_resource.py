from unittest import mock
from heat.engine.resources.openstack.neutron.taas import tap_flow
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_show_resource(self):
    mock_tap_flow_get = self.test_client_plugin.show_ext_resource
    mock_tap_flow_get.return_value = {}
    self.assertEqual({}, self.test_resource._show_resource(), 'Failed to show resource')