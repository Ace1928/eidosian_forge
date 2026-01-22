import ctypes
import functools
import inspect
import socket
import time
from oslo_log import log as logging
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def _get_iscsi_device_from_session(self, session_id, target_lun):
    devices = self._get_iscsi_session_devices(session_id)
    for device in devices:
        if device.ScsiAddress.Lun == target_lun:
            return device