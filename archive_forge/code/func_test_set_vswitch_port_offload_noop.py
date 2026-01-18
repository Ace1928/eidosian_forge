from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_hw_offload_sd_from_port_alloc')
def test_set_vswitch_port_offload_noop(self, mock_get_hw_offload_sd):
    self._mock_get_switch_port_alloc()
    self.netutils.set_vswitch_port_offload(mock.sentinel.port_name)
    self.netutils._jobutils.modify_virt_feature.assert_not_called()