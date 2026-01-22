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
@mock.patch.object(vhdutils.VHDUtils, 'get_vhd_info')
@mock.patch.object(vhdutils.VHDUtils, '_get_internal_vhd_size_by_file_size')
@mock.patch.object(vhdutils.VHDUtils, '_get_internal_vhdx_size_by_file_size')
def _test_get_int_sz_by_file_size(self, mock_get_vhdx_int_size, mock_get_vhd_int_size, mock_get_vhd_info, vhd_dev_id=w_const.VIRTUAL_STORAGE_TYPE_DEVICE_VHD, vhd_type=constants.VHD_TYPE_DYNAMIC):
    fake_vhd_info = dict(ProviderSubtype=vhd_type, ParentPath=mock.sentinel.parent_path, DeviceId=vhd_dev_id)
    mock_get_vhd_info.side_effect = [fake_vhd_info]
    exppected_vhd_info_calls = [mock.call(mock.sentinel.vhd_path)]
    expected_vhd_checked = mock.sentinel.vhd_path
    expected_checked_vhd_info = fake_vhd_info
    if vhd_type == constants.VHD_TYPE_DIFFERENCING:
        expected_checked_vhd_info = dict(fake_vhd_info, vhd_type=constants.VHD_TYPE_DYNAMIC)
        mock_get_vhd_info.side_effect.append(expected_checked_vhd_info)
        exppected_vhd_info_calls.append(mock.call(mock.sentinel.parent_path))
        expected_vhd_checked = mock.sentinel.parent_path
    is_vhd = vhd_dev_id == w_const.VIRTUAL_STORAGE_TYPE_DEVICE_VHD
    expected_helper = mock_get_vhd_int_size if is_vhd else mock_get_vhdx_int_size
    ret_val = self._vhdutils.get_internal_vhd_size_by_file_size(mock.sentinel.vhd_path, mock.sentinel.vhd_size)
    mock_get_vhd_info.assert_has_calls(exppected_vhd_info_calls)
    expected_helper.assert_called_once_with(expected_vhd_checked, mock.sentinel.vhd_size, expected_checked_vhd_info)
    self.assertEqual(expected_helper.return_value, ret_val)