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
@mock.patch.object(vhdutils.VHDUtils, 'get_internal_vhd_size_by_file_size')
@mock.patch.object(vhdutils.VHDUtils, '_resize_vhd')
@mock.patch.object(vhdutils.VHDUtils, '_check_resize_needed')
def _test_resize_vhd(self, mock_check_resize_needed, mock_resize_helper, mock_get_internal_size, is_file_max_size=True, resize_needed=True):
    mock_check_resize_needed.return_value = resize_needed
    self._vhdutils.resize_vhd(mock.sentinel.vhd_path, mock.sentinel.new_size, is_file_max_size, validate_new_size=True)
    if is_file_max_size:
        mock_get_internal_size.assert_called_once_with(mock.sentinel.vhd_path, mock.sentinel.new_size)
        expected_new_size = mock_get_internal_size.return_value
    else:
        expected_new_size = mock.sentinel.new_size
    mock_check_resize_needed.assert_called_once_with(mock.sentinel.vhd_path, expected_new_size)
    if resize_needed:
        mock_resize_helper.assert_called_once_with(mock.sentinel.vhd_path, expected_new_size)
    else:
        self.assertFalse(mock_resize_helper.called)