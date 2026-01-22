from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_set_boot_order_gen1')
@mock.patch.object(vmutils.VMUtils, '_set_boot_order_gen2')
@mock.patch.object(vmutils.VMUtils, 'get_vm_generation')
def _test_set_boot_order(self, mock_get_vm_gen, mock_set_boot_order_gen2, mock_set_boot_order_gen1, vm_gen):
    mock_get_vm_gen.return_value = vm_gen
    self._vmutils.set_boot_order(mock.sentinel.fake_vm_name, mock.sentinel.boot_order)
    if vm_gen == constants.VM_GEN_1:
        mock_set_boot_order_gen1.assert_called_once_with(mock.sentinel.fake_vm_name, mock.sentinel.boot_order)
    else:
        mock_set_boot_order_gen2.assert_called_once_with(mock.sentinel.fake_vm_name, mock.sentinel.boot_order)