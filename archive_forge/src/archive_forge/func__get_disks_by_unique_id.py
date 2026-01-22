import ctypes
import os
import re
import threading
from collections.abc import Iterable
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import pathutils
from os_win.utils import win32utils
from os_win.utils.winapi import libs as w_lib
def _get_disks_by_unique_id(self, unique_id, unique_id_format):
    disks = self._conn_storage.Msft_Disk(UniqueId=unique_id, UniqueIdFormat=unique_id_format)
    if not disks:
        err_msg = _("Could not find any disk having unique id '%(unique_id)s' and unique id format '%(unique_id_format)s'")
        raise exceptions.DiskNotFound(err_msg % dict(unique_id=unique_id, unique_id_format=unique_id_format))
    return disks