from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(win32utils.Win32Utils, 'run_and_check_output')
def test_wait_for_multiple_objects(self, mock_helper):
    fake_handles = [10, 11]
    ret_val = self._win32_utils.wait_for_multiple_objects(fake_handles, mock.sentinel.wait_all, mock.sentinel.milliseconds)
    mock_helper.assert_called_once_with(win32utils.kernel32.WaitForMultipleObjects, len(fake_handles), mock.ANY, mock.sentinel.wait_all, mock.sentinel.milliseconds, kernel32_lib_func=True, error_ret_vals=[w_const.WAIT_FAILED])
    self.assertEqual(mock_helper.return_value, ret_val)
    handles_arg = mock_helper.call_args_list[0][0][2]
    self.assertIsInstance(handles_arg, wintypes.HANDLE * len(fake_handles))
    self.assertEqual(fake_handles, handles_arg[:])