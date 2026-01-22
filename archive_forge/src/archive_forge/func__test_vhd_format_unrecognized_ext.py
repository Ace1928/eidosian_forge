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
@mock.patch.object(vhdutils.VHDUtils, '_get_vhd_format_by_signature')
@mock.patch('os.path.exists')
def _test_vhd_format_unrecognized_ext(self, mock_exists, mock_get_vhd_fmt_by_sign, signature_available=False):
    mock_exists.return_value = True
    fake_vhd_path = 'C:\\test_vhd'
    mock_get_vhd_fmt_by_sign.return_value = constants.DISK_FORMAT_VHD if signature_available else None
    if signature_available:
        ret_val = self._vhdutils.get_vhd_format(fake_vhd_path)
        self.assertEqual(constants.DISK_FORMAT_VHD, ret_val)
    else:
        self.assertRaises(exceptions.VHDException, self._vhdutils.get_vhd_format, fake_vhd_path)