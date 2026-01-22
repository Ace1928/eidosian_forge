from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def _lookup_vm(self):
    mock_vm = mock.MagicMock()
    self._vmutils._lookup_vm_check = mock.MagicMock(return_value=mock_vm)
    mock_vm.path_.return_value = self._FAKE_VM_PATH
    return mock_vm