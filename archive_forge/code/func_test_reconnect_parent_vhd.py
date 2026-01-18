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
def test_reconnect_parent_vhd(self, mock_open):
    set_vdisk_info_struct = self._vdisk_struct.SET_VIRTUAL_DISK_INFO
    open_params_struct = self._vdisk_struct.OPEN_VIRTUAL_DISK_PARAMETERS
    fake_set_params = set_vdisk_info_struct.return_value
    fake_open_params = open_params_struct.return_value
    mock_open.return_value = mock.sentinel.handle
    self._vhdutils.reconnect_parent_vhd(mock.sentinel.vhd_path, mock.sentinel.parent_path)
    self.assertEqual(w_const.OPEN_VIRTUAL_DISK_VERSION_2, fake_open_params.Version)
    self.assertFalse(fake_open_params.Version2.GetInfoOnly)
    self._vhdutils._open.assert_called_once_with(mock.sentinel.vhd_path, open_flag=w_const.OPEN_VIRTUAL_DISK_FLAG_NO_PARENTS, open_access_mask=0, open_params=vhdutils.ctypes.byref(fake_open_params))
    self.assertEqual(w_const.SET_VIRTUAL_DISK_INFO_PARENT_PATH, fake_set_params.Version)
    self.assertEqual(mock.sentinel.parent_path, fake_set_params.ParentFilePath)
    self._mock_run.assert_called_once_with(vhdutils.virtdisk.SetVirtualDiskInformation, mock.sentinel.handle, vhdutils.ctypes.byref(fake_set_params), **self._run_args)
    self._mock_close.assert_called_once_with(mock.sentinel.handle)