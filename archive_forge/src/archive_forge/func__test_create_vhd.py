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
@mock.patch.object(vhdutils.VHDUtils, '_get_vhd_device_id')
def _test_create_vhd(self, mock_get_dev_id, new_vhd_type):
    create_params_struct = self._vdisk_struct.CREATE_VIRTUAL_DISK_PARAMETERS
    mock_handle = vhdutils.wintypes.HANDLE.return_value
    fake_vst = self._fake_vst_struct.return_value
    fake_create_params = create_params_struct.return_value
    expected_create_vhd_flag = vhdutils.CREATE_VIRTUAL_DISK_FLAGS.get(new_vhd_type, 0)
    self._vhdutils.create_vhd(new_vhd_path=mock.sentinel.new_vhd_path, new_vhd_type=new_vhd_type, src_path=mock.sentinel.src_path, max_internal_size=mock.sentinel.max_internal_size, parent_path=mock.sentinel.parent_path, guid=mock.sentinel.guid)
    self._fake_vst_struct.assert_called_once_with(DeviceId=mock_get_dev_id.return_value, VendorId=w_const.VIRTUAL_STORAGE_TYPE_VENDOR_MICROSOFT)
    self.assertEqual(w_const.CREATE_VIRTUAL_DISK_VERSION_2, fake_create_params.Version)
    self.assertEqual(mock.sentinel.max_internal_size, fake_create_params.Version2.MaximumSize)
    self.assertEqual(mock.sentinel.parent_path, fake_create_params.Version2.ParentPath)
    self.assertEqual(mock.sentinel.src_path, fake_create_params.Version2.SourcePath)
    self.assertEqual(vhdutils.VIRTUAL_DISK_DEFAULT_PHYS_SECTOR_SIZE, fake_create_params.Version2.PhysicalSectorSizeInBytes)
    self.assertEqual(w_const.CREATE_VHD_PARAMS_DEFAULT_BLOCK_SIZE, fake_create_params.Version2.BlockSizeInBytes)
    self.assertEqual(vhdutils.VIRTUAL_DISK_DEFAULT_SECTOR_SIZE, fake_create_params.Version2.SectorSizeInBytes)
    self.assertEqual(vhdutils.wintypes.GUID.from_str.return_value, fake_create_params.Version2.UniqueId)
    vhdutils.wintypes.GUID.from_str.assert_called_once_with(mock.sentinel.guid)
    self._mock_run.assert_called_once_with(vhdutils.virtdisk.CreateVirtualDisk, self._ctypes.byref(fake_vst), self._ctypes.c_wchar_p(mock.sentinel.new_vhd_path), 0, None, expected_create_vhd_flag, 0, self._ctypes.byref(fake_create_params), None, self._ctypes.byref(mock_handle), **self._run_args)
    self._mock_close.assert_called_once_with(mock_handle)