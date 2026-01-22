from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def _mock_get_switch_port_alloc(self, found=True):
    mock_port = mock.MagicMock()
    patched = mock.patch.object(self.netutils, '_get_switch_port_allocation', return_value=(mock_port, found))
    patched.start()
    self.addCleanup(patched.stop)
    return mock_port