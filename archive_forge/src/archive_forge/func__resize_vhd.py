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
def _resize_vhd(self, vhd_path, new_max_size):
    handle = self._open(vhd_path)
    params = vdisk_struct.RESIZE_VIRTUAL_DISK_PARAMETERS()
    params.Version = w_const.RESIZE_VIRTUAL_DISK_VERSION_1
    params.Version1.NewSize = new_max_size
    self._run_and_check_output(virtdisk.ResizeVirtualDisk, handle, 0, ctypes.byref(params), None, cleanup_handle=handle)