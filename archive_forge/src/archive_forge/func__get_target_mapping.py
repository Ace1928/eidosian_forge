import contextlib
import ctypes
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import hbaapi as fc_struct
def _get_target_mapping(self, hba_handle):
    entry_count = 0
    hba_status = HBA_STATUS_ERROR_MORE_DATA
    while hba_status == HBA_STATUS_ERROR_MORE_DATA:
        mapping = fc_struct.get_target_mapping_struct(entry_count)
        hba_status = self._run_and_check_output(hbaapi.HBA_GetFcpTargetMapping, hba_handle, ctypes.byref(mapping), ignored_error_codes=[HBA_STATUS_ERROR_MORE_DATA])
        entry_count = mapping.NumberOfEntries
    return mapping