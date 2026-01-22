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
def _get_adapter_name(self, adapter_index):
    buff = (ctypes.c_char * w_const.MAX_ISCSI_HBANAME_LEN)()
    self._run_and_check_output(hbaapi.HBA_GetAdapterName, ctypes.c_uint32(adapter_index), buff)
    return buff.value.decode('utf-8')