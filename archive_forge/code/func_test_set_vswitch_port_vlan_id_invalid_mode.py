from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_set_vswitch_port_vlan_id_invalid_mode(self):
    self.assertRaises(AttributeError, self.netutils.set_vswitch_port_vlan_id, mock.sentinel.vlan_id, mock.sentinel.switch_port_name, operation_mode=mock.sentinel.invalid_mode)