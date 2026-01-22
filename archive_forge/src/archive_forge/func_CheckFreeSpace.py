from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import locale
import os
import struct
import sys
import six
from gslib.utils.constants import WINDOWS_1252
def CheckFreeSpace(path):
    """Return path/drive free space (in bytes)."""
    if IS_WINDOWS:
        try:
            get_disk_free_space_ex = WINFUNCTYPE(c_int, c_wchar_p, POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64))
            get_disk_free_space_ex = get_disk_free_space_ex(('GetDiskFreeSpaceExW', windll.kernel32), ((1, 'lpszPathName'), (2, 'lpFreeUserSpace'), (2, 'lpTotalSpace'), (2, 'lpFreeSpace')))
        except AttributeError:
            get_disk_free_space_ex = WINFUNCTYPE(c_int, c_char_p, POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64))
            get_disk_free_space_ex = get_disk_free_space_ex(('GetDiskFreeSpaceExA', windll.kernel32), ((1, 'lpszPathName'), (2, 'lpFreeUserSpace'), (2, 'lpTotalSpace'), (2, 'lpFreeSpace')))

        def GetDiskFreeSpaceExErrCheck(result, unused_func, args):
            if not result:
                raise WinError()
            return args[1].value
        get_disk_free_space_ex.errcheck = GetDiskFreeSpaceExErrCheck
        return get_disk_free_space_ex(os.getenv('SystemDrive'))
    else:
        _, f_frsize, _, _, f_bavail, _, _, _, _, _ = os.statvfs(path)
        return f_frsize * f_bavail