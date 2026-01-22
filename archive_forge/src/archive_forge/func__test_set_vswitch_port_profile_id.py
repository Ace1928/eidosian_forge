from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_prepare_profile_sd')
@mock.patch.object(networkutils.NetworkUtils, '_get_profile_setting_data_from_port_alloc')
def _test_set_vswitch_port_profile_id(self, mock_get_profile_setting_data_from_port_alloc, mock_prepare_profile_sd, found, side_effect=None):
    mock_port_profile = mock.MagicMock()
    mock_new_port_profile = mock.MagicMock()
    mock_port_alloc = self._mock_get_switch_port_alloc()
    mock_add_feature = self.netutils._jobutils.add_virt_feature
    mock_remove_feature = self.netutils._jobutils.remove_virt_feature
    mock_get_profile_setting_data_from_port_alloc.return_value = mock_port_profile if found else None
    mock_prepare_profile_sd.return_value = mock_new_port_profile
    mock_add_feature.side_effect = side_effect
    fake_params = {'switch_port_name': self._FAKE_PORT_NAME, 'profile_id': mock.sentinel.profile_id, 'profile_data': mock.sentinel.profile_data, 'profile_name': mock.sentinel.profile_name, 'net_cfg_instance_id': None, 'cdn_label_id': None, 'cdn_label_string': None, 'vendor_id': None, 'vendor_name': mock.sentinel.vendor_name}
    if side_effect:
        self.assertRaises(exceptions.HyperVException, self.netutils.set_vswitch_port_profile_id, **fake_params)
    else:
        self.netutils.set_vswitch_port_profile_id(**fake_params)
    fake_params.pop('switch_port_name')
    mock_prepare_profile_sd.assert_called_once_with(**fake_params)
    if found:
        mock_remove_feature.assert_called_once_with(mock_port_profile)
        self.assertNotIn(self._FAKE_INSTANCE_ID, self.netutils._profile_sds)
    mock_get_profile_setting_data_from_port_alloc.assert_called_with(mock_port_alloc)
    self.assertNotIn(mock_port_alloc, self.netutils._profile_sds)
    mock_add_feature.assert_called_once_with(mock_new_port_profile, mock_port_alloc)