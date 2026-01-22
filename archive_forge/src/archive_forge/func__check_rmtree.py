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
@mock.patch('time.sleep')
@mock.patch.object(pathutils.shutil, 'rmtree')
def _check_rmtree(self, mock_rmtree, mock_sleep, side_effect):
    mock_rmtree.side_effect = side_effect
    self.assertRaises(exceptions.WindowsError, self._pathutils.rmtree, mock.sentinel.FAKE_PATH)