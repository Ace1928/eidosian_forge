from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_remove_virt_resource(self):
    self._test_virt_method('RemoveResourceSettings', 2, 'remove_virt_resource', False, ResourceSettings=[mock.sentinel.res_path])