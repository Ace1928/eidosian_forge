from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def _test_get_vm_generation(self, vm_gen):
    mock_settings = self._lookup_vm()
    vm_gen_string = 'Microsoft:Hyper-V:SubType:' + str(vm_gen)
    mock_settings.VirtualSystemSubType = vm_gen_string
    ret = self._vmutils.get_vm_generation(mock.sentinel.FAKE_VM_NAME)
    self.assertEqual(vm_gen, ret)