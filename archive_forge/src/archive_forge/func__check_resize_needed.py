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
def _check_resize_needed(self, vhd_path, new_size):
    curr_size = self.get_vhd_size(vhd_path)['VirtualSize']
    if curr_size > new_size:
        err_msg = _('Cannot resize image %(vhd_path)s to a smaller size. Image virtual size: %(curr_size)s, Requested virtual size: %(new_size)s')
        raise exceptions.VHDException(err_msg % dict(vhd_path=vhd_path, curr_size=curr_size, new_size=new_size))
    elif curr_size == new_size:
        LOG.debug('Skipping resizing %(vhd_path)s to %(new_size)sas it already has the requested size.', dict(vhd_path=vhd_path, new_size=new_size))
        return False
    return True