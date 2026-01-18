import ctypes
import os
import shutil
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import pathutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
@mock.patch('time.time')
def test_rmtree_exceeded(self, mock_time):
    mock_time.side_effect = range(1, 100, 10)
    exc = exceptions.WindowsError()
    exc.winerror = w_const.ERROR_DIR_IS_NOT_EMPTY
    self._check_rmtree(side_effect=exc)