from unittest import mock
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron.sfc import port_pair_group
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_handle_update_port_pairs(self):
    self.patchobject(self.test_client_plugin, 'resolve_ext_resource').return_value = 'port2'
    mock_ppg_patch = self.test_client_plugin.update_ext_resource
    self.test_resource.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {port_pair_group.PortPairGroup.NAME: 'name', port_pair_group.PortPairGroup.DESCRIPTION: 'description', port_pair_group.PortPairGroup.PORT_PAIRS: ['port2']}
    self.test_resource.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    mock_ppg_patch.assert_called_once_with('port_pair_group', {'name': 'name', 'description': 'description', 'port_pairs': ['port2']}, self.test_resource.resource_id)