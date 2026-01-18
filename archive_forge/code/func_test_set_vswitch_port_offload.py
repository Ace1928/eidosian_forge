from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_hw_offload_sd_from_port_alloc')
def test_set_vswitch_port_offload(self, mock_get_hw_offload_sd):
    mock_port_alloc = self._mock_get_switch_port_alloc()
    mock_hw_offload_sd = mock_get_hw_offload_sd.return_value
    iov_queues = 1
    offloaded_sa = 1
    self.netutils.set_vswitch_port_offload(mock.sentinel.port_name, True, iov_queues, True, offloaded_sa)
    mock_get_hw_offload_sd.assert_called_once_with(mock_port_alloc)
    self.netutils._jobutils.modify_virt_feature.assert_called_with(mock_hw_offload_sd)
    self.assertEqual(self.netutils._OFFLOAD_ENABLED, mock_hw_offload_sd.IOVOffloadWeight)
    self.assertEqual(iov_queues, mock_hw_offload_sd.IOVQueuePairsRequested)
    self.assertEqual(self.netutils._OFFLOAD_ENABLED, mock_hw_offload_sd.VMQOffloadWeight)
    self.assertEqual(offloaded_sa, mock_hw_offload_sd.IPSecOffloadLimit)