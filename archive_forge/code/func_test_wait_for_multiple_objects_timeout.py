from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(win32utils.Win32Utils, 'run_and_check_output')
def test_wait_for_multiple_objects_timeout(self, mock_helper):
    fake_handles = [10]
    mock_helper.return_value = w_const.ERROR_WAIT_TIMEOUT
    self.assertRaises(exceptions.Timeout, self._win32_utils.wait_for_multiple_objects, fake_handles, mock.sentinel.wait_all, mock.sentinel.milliseconds)