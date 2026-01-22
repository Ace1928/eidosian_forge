from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_prepare_vlan_sd_trunk_mode')
@mock.patch.object(networkutils.NetworkUtils, '_prepare_vlan_sd_access_mode')
def _check_set_vswitch_port_vlan_id(self, mock_prepare_vlan_sd_access, mock_prepare_vlan_sd_trunk, op_mode=constants.VLAN_MODE_ACCESS, missing_vlan=False):
    mock_port = self._mock_get_switch_port_alloc(found=True)
    old_vlan_settings = mock.MagicMock()
    if missing_vlan:
        side_effect = [old_vlan_settings, None]
    else:
        side_effect = [old_vlan_settings, old_vlan_settings]
    self.netutils._get_vlan_setting_data_from_port_alloc = mock.MagicMock(side_effect=side_effect)
    mock_vlan_settings = mock.MagicMock()
    mock_prepare_vlan_sd_access.return_value = mock_vlan_settings
    mock_prepare_vlan_sd_trunk.return_value = mock_vlan_settings
    if missing_vlan:
        self.assertRaises(exceptions.HyperVException, self.netutils.set_vswitch_port_vlan_id, self._FAKE_VLAN_ID, self._FAKE_PORT_NAME, operation_mode=op_mode)
    else:
        self.netutils.set_vswitch_port_vlan_id(self._FAKE_VLAN_ID, self._FAKE_PORT_NAME, operation_mode=op_mode)
    if op_mode == constants.VLAN_MODE_ACCESS:
        mock_prepare_vlan_sd_access.assert_called_once_with(old_vlan_settings, self._FAKE_VLAN_ID)
    else:
        mock_prepare_vlan_sd_trunk.assert_called_once_with(old_vlan_settings, self._FAKE_VLAN_ID, None)
    mock_remove_feature = self.netutils._jobutils.remove_virt_feature
    mock_remove_feature.assert_called_once_with(old_vlan_settings)
    mock_add_feature = self.netutils._jobutils.add_virt_feature
    mock_add_feature.assert_called_once_with(mock_vlan_settings, mock_port)