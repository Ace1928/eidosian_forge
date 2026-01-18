from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import processutils
from os_win.utils.winapi import constants as w_const
def test_release_mutex(self):
    self._procutils.release_mutex(mock.sentinel.handle)
    self._mock_run.assert_called_once_with(self._mock_kernel32.ReleaseMutex, mock.sentinel.handle, kernel32_lib_func=True)