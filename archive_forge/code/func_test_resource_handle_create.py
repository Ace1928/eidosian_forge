from unittest import mock
from heat.engine.resources.openstack.neutron.taas import tap_flow
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_handle_create(self):
    mock_tap_flow_create = self.test_client_plugin.create_ext_resource
    mock_resource = self._get_mock_resource()
    mock_tap_flow_create.return_value = mock_resource
    self.assertEqual('test_tap_flow', self.test_resource.properties.get(tap_flow.TapFlow.NAME))
    self.assertEqual('desc', self.test_resource.properties.get(tap_flow.TapFlow.DESCRIPTION))
    self.assertEqual('6af055d3-26f6-48dd-a597-7611d7e58d35', self.test_resource.properties.get(tap_flow.TapFlow.PORT))
    self.assertEqual('6af055d3-26f6-48dd-a597-7611d7e58d35', self.test_resource.properties.get(tap_flow.TapFlow.TAP_SERVICE))
    self.assertEqual('BOTH', self.test_resource.properties.get(tap_flow.TapFlow.DIRECTION))
    self.assertEqual('1-5,9,18,27-30,99-108,4000-4095', self.test_resource.properties.get(tap_flow.TapFlow.VLAN_FILTER))
    self.test_resource.data_set = mock.Mock()
    self.test_resource.handle_create()
    mock_tap_flow_create.assert_called_once_with('tap_flow', {'name': 'test_tap_flow', 'description': 'desc', 'port': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'tap_service': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'direction': 'BOTH', 'vlan_filter': '1-5,9,18,27-30,99-108,4000-4095'})