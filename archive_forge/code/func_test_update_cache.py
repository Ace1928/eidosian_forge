from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_update_cache(self):
    self.netutils._enable_cache = True
    self.netutils._switch_ports[mock.sentinel.other] = mock.sentinel.port
    conn = self.netutils._conn
    mock_port = mock.MagicMock(ElementName=mock.sentinel.port_name)
    conn.Msvm_EthernetPortAllocationSettingData.return_value = [mock_port]
    self.netutils.update_cache()
    self.assertEqual({mock.sentinel.port_name: mock_port}, self.netutils._switch_ports)
    netutils = networkutils.NetworkUtils()
    self.assertEqual({mock.sentinel.port_name: mock_port}, netutils._switch_ports)