from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_update_cache_disabled(self):
    self.netutils._enable_cache = False
    self.netutils._switch_ports = {}
    self.netutils.update_cache()
    conn = self.netutils._conn
    conn.Msvm_EthernetPortAllocationSettingData.assert_not_called()
    self.assertEqual({}, self.netutils._switch_ports)