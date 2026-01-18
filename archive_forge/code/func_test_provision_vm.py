from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_provision_vm(self):
    self.assertRaises(NotImplementedError, self._vmutils.provision_vm, mock.sentinel.vm_name, mock.sentinel.fsk_filepath, mock.sentinel.pdk_filepath)