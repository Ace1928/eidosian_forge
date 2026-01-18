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
@mock.patch('oslo_utils.fileutils.delete_if_exists')
def test_temporary_file(self, mock_delete):
    self._pathutils.create_temporary_file = mock.MagicMock()
    self._pathutils.create_temporary_file.return_value = mock.sentinel.temporary_file
    with self._pathutils.temporary_file() as tmp_file:
        self.assertEqual(mock.sentinel.temporary_file, tmp_file)
        self.assertFalse(mock_delete.called)
    mock_delete.assert_called_once_with(mock.sentinel.temporary_file)