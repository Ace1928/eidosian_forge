import ctypes
import os
from unittest import mock
import uuid
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.virtdisk import vhdutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(vhdutils.VHDUtils, '_open')
def test_resize_vhd_helper(self, mock_open):
    resize_vdisk_struct = self._vdisk_struct.RESIZE_VIRTUAL_DISK_PARAMETERS
    fake_params = resize_vdisk_struct.return_value
    mock_open.return_value = mock.sentinel.handle
    self._vhdutils._resize_vhd(mock.sentinel.vhd_path, mock.sentinel.new_size)
    self.assertEqual(w_const.RESIZE_VIRTUAL_DISK_VERSION_1, fake_params.Version)
    self.assertEqual(mock.sentinel.new_size, fake_params.Version1.NewSize)
    self._mock_run.assert_called_once_with(vhdutils.virtdisk.ResizeVirtualDisk, mock.sentinel.handle, 0, vhdutils.ctypes.byref(fake_params), None, **self._run_args)
    self._mock_close.assert_called_once_with(mock.sentinel.handle)