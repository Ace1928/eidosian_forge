from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_prepare_vlan_sd_access_mode')
def test_set_vswitch_port_vlan_id_already_set(self, mock_prepare_vlan_sd):
    self._mock_get_switch_port_alloc()
    mock_prepare_vlan_sd.return_value = None
    self.netutils.set_vswitch_port_vlan_id(mock.sentinel.vlan_id, mock.sentinel.port_name)
    mock_remove_feature = self.netutils._jobutils.remove_virt_feature
    self.assertFalse(mock_remove_feature.called)