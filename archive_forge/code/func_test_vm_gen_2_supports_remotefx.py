from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_vm_gen_2_supports_remotefx(self):
    ret = self._vmutils.vm_gen_supports_remotefx(constants.VM_GEN_2)
    self.assertFalse(ret)