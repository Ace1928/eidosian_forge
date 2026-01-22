import ctypes
import os
import struct
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import virtdisk as vdisk_struct
from os_win.utils.winapi import wintypes
def _get_vhd_format_by_signature(self, vhd_path):
    with open(vhd_path, 'rb') as f:
        if f.read(8) == VHDX_SIGNATURE:
            return constants.DISK_FORMAT_VHDX
        f.seek(0, 2)
        file_size = f.tell()
        if file_size >= 512:
            f.seek(-512, 2)
            if f.read(8) == VHD_SIGNATURE:
                return constants.DISK_FORMAT_VHD