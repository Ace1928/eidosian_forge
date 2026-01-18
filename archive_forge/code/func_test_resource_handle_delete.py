from unittest import mock
from heat.engine.resources.openstack.neutron.taas import tap_flow
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_handle_delete(self):
    mock_tap_flow_delete = self.test_client_plugin.delete_ext_resource
    self.test_resource.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    mock_tap_flow_delete.return_value = None
    self.assertIsNone(self.test_resource.handle_delete())
    mock_tap_flow_delete.assert_called_once_with('tap_flow', self.test_resource.resource_id)