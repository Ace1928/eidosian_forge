from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_set_empty_port_qos_rule(self):
    self._mock_get_switch_port_alloc()
    self.netutils.set_port_qos_rule(mock.sentinel.port_id, {})
    self.assertFalse(self.netutils._get_switch_port_allocation.called)