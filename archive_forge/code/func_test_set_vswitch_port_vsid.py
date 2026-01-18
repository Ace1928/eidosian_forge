from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_set_switch_port_security_settings')
def test_set_vswitch_port_vsid(self, mock_set_port_sec_settings):
    self.netutils.set_vswitch_port_vsid(mock.sentinel.vsid, mock.sentinel.switch_port_name)
    mock_set_port_sec_settings.assert_called_once_with(mock.sentinel.switch_port_name, VirtualSubnetId=mock.sentinel.vsid)