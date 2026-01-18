from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_bandwidth_setting_data_from_port_alloc')
@mock.patch.object(networkutils.NetworkUtils, '_get_default_setting_data')
def test_set_port_qos_rule_hyperv_exc(self, mock_get_default_sd, mock_get_bandwidth_sd):
    mock_port_alloc = self._mock_get_switch_port_alloc()
    self.netutils._bandwidth_sds = {mock_port_alloc.InstanceID: mock.sentinel.InstanceID}
    mock_remove_feature = self.netutils._jobutils.remove_virt_feature
    mock_add_feature = self.netutils._jobutils.add_virt_feature
    mock_add_feature.side_effect = exceptions.HyperVException
    qos_rule = dict(min_kbps=20000, max_kbps=30000, max_burst_kbps=40000, max_burst_size_kb=50000)
    self.assertRaises(exceptions.HyperVException, self.netutils.set_port_qos_rule, mock.sentinel.port_id, qos_rule)
    mock_get_bandwidth_sd.assert_called_once_with(mock_port_alloc)
    mock_get_default_sd.assert_called_once_with(self.netutils._PORT_BANDWIDTH_SET_DATA)
    mock_remove_feature.assert_called_once_with(mock_get_bandwidth_sd.return_value)
    mock_add_feature.assert_called_once_with(mock_get_default_sd.return_value, mock_port_alloc)
    bw = mock_get_default_sd.return_value
    self.assertEqual(qos_rule['min_kbps'] * units.Ki, bw.Reservation)
    self.assertEqual(qos_rule['max_kbps'] * units.Ki, bw.Limit)
    self.assertEqual(qos_rule['max_burst_kbps'] * units.Ki, bw.BurstLimit)
    self.assertEqual(qos_rule['max_burst_size_kb'] * units.Ki, bw.BurstSize)
    self.assertNotIn(mock_port_alloc.InstanceID, self.netutils._bandwidth_sds)