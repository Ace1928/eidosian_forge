from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import processutils
from os_win.utils.winapi import constants as w_const
def test_open_process(self):
    process_handle = self._procutils.open_process(mock.sentinel.pid, mock.sentinel.desired_access, mock.sentinel.inherit_handle)
    self._mock_run.assert_called_once_with(self._mock_kernel32.OpenProcess, mock.sentinel.desired_access, mock.sentinel.inherit_handle, mock.sentinel.pid, error_ret_vals=[None], kernel32_lib_func=True)
    self.assertEqual(self._mock_run.return_value, process_handle)