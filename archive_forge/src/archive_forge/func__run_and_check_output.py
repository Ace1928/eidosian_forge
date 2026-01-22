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
def _run_and_check_output(self, *args, **kwargs):
    cleanup_handle = kwargs.pop('cleanup_handle', None)
    kwargs.update(self._virtdisk_run_args)
    try:
        return self._win32_utils.run_and_check_output(*args, **kwargs)
    finally:
        if cleanup_handle:
            self._win32_utils.close_handle(cleanup_handle)