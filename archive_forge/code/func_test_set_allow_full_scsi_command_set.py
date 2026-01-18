from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_modify_virtual_system')
def test_set_allow_full_scsi_command_set(self, mock_modify_virtual_system):
    mock_vm = self._lookup_vm()
    self._vmutils.enable_vm_full_scsi_command_set(mock.sentinel.vm_name)
    self.assertTrue(mock_vm.AllowFullSCSICommandSet)
    mock_modify_virtual_system.assert_called_once_with(mock_vm)