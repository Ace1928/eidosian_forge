from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
def test_vm_has_s3_controller_gen1(self):
    self.assertTrue(self._test_vm_has_s3_controller(constants.VM_GEN_1))