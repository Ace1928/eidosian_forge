from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(_utils, 'avoid_blocking_call')
def test_run_and_check_output_eventlet_nb_mode_enabled(self, mock_helper):
    self._win32_utils.run_and_check_output(mock.sentinel.func, mock.sentinel.arg, eventlet_nonblocking_mode=True)
    mock_helper.assert_called_once_with(self._win32_utils._run_and_check_output, mock.sentinel.func, mock.sentinel.arg)