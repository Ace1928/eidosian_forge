from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_create_default_setting_data')
def test_prepare_vlan_sd_trunk_mode(self, mock_create_default_sd):
    mock_vlan_sd = mock_create_default_sd.return_value
    actual_vlan_sd = self.netutils._prepare_vlan_sd_trunk_mode(None, mock.sentinel.vlan_id, mock.sentinel.trunk_vlans)
    self.assertEqual(mock_vlan_sd, actual_vlan_sd)
    self.assertEqual(mock.sentinel.vlan_id, mock_vlan_sd.NativeVlanId)
    self.assertEqual(mock.sentinel.trunk_vlans, mock_vlan_sd.TrunkVlanIdArray)
    self.assertEqual(constants.VLAN_MODE_TRUNK, mock_vlan_sd.OperationMode)
    mock_create_default_sd.assert_called_once_with(self.netutils._PORT_VLAN_SET_DATA)