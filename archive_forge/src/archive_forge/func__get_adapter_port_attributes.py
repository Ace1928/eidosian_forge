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
def _get_adapter_port_attributes(self, hba_handle, port_index):
    port_attributes = fc_struct.HBA_PortAttributes()
    self._run_and_check_output(hbaapi.HBA_GetAdapterPortAttributes, hba_handle, port_index, ctypes.byref(port_attributes))
    return port_attributes