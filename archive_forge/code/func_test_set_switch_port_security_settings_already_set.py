from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_security_setting_data_from_port_alloc')
def test_set_switch_port_security_settings_already_set(self, mock_get_sec_sd):
    self._mock_get_switch_port_alloc()
    mock_sec_sd = mock.MagicMock(VirtualSubnetId=mock.sentinel.vsid, AllowMacSpoofing=mock.sentinel.state)
    mock_get_sec_sd.return_value = mock_sec_sd
    self.netutils._set_switch_port_security_settings(mock.sentinel.switch_port_name, VirtualSubnetId=mock.sentinel.vsid, AllowMacSpoofing=mock.sentinel.state)
    self.assertFalse(self.netutils._jobutils.remove_virt_feature.called)
    self.assertFalse(self.netutils._jobutils.add_virt_feature.called)