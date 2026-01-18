from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_soft_shutdown_vm_wmi_exc(self):
    self._lookup_vm()
    mock_shutdown = mock.MagicMock()
    mock_shutdown.InitiateShutdown.side_effect = exceptions.x_wmi
    self._vmutils._conn.Msvm_ShutdownComponent.return_value = [mock_shutdown]
    self.assertRaises(exceptions.HyperVException, self._vmutils.soft_shutdown_vm, self._FAKE_VM_NAME)