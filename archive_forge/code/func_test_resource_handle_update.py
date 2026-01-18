from unittest import mock
from heat.engine.resources.openstack.neutron.taas import tap_flow
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_handle_update(self):
    mock_tap_flow_patch = self.test_client_plugin.update_ext_resource
    self.test_resource.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {tap_flow.TapFlow.NAME: 'name-updated', tap_flow.TapFlow.DESCRIPTION: 'description-updated', tap_flow.TapFlow.PORT: '6af055d3-26f6-48dd-a597-7611d7e58d35', tap_flow.TapFlow.TAP_SERVICE: '6af055d3-26f6-48dd-a597-7611d7e58d35', tap_flow.TapFlow.DIRECTION: 'BOTH', tap_flow.TapFlow.VLAN_FILTER: '1-5,9,18,27-30,99-108,4000-4095'}
    self.test_resource.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    mock_tap_flow_patch.assert_called_once_with('tap_flow', {'name': 'name-updated', 'description': 'description-updated', 'port': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'tap_service': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'direction': 'BOTH', 'vlan_filter': '1-5,9,18,27-30,99-108,4000-4095'}, self.test_resource.resource_id)