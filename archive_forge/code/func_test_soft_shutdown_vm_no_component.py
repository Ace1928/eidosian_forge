from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_soft_shutdown_vm_no_component(self):
    mock_vm = self._lookup_vm()
    self._vmutils._conn.Msvm_ShutdownComponent.return_value = []
    self._vmutils.soft_shutdown_vm(self._FAKE_VM_NAME)
    self._vmutils._conn.Msvm_ShutdownComponent.assert_called_once_with(SystemName=mock_vm.Name)
    self.assertFalse(self._vmutils._jobutils.check_ret_val.called)