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
def _get_vhdx_metadata_size_and_offset(self, vhdx_file):
    offset = VHDX_METADATA_SIZE_OFFSET + VHDX_REGION_TABLE_OFFSET
    vhdx_file.seek(offset)
    metadata_offset = struct.unpack('<Q', vhdx_file.read(8))[0]
    metadata_size = struct.unpack('<I', vhdx_file.read(4))[0]
    return (metadata_size, metadata_offset)