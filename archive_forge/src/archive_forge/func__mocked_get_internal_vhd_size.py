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
def _mocked_get_internal_vhd_size(self, root_vhd_size, vhd_type):
    fake_vhd_info = dict(ProviderSubtype=vhd_type, BlockSize=2097152, ParentPath=mock.sentinel.parent_path)
    return self._vhdutils._get_internal_vhd_size_by_file_size(mock.sentinel.vhd_path, root_vhd_size, fake_vhd_info)