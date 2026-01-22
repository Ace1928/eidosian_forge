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
@retry_decorator(error_codes=w_const.ERROR_INSUFFICIENT_BUFFER)
def _login_iscsi_target(self, target_name, portal=None, login_opts=None, is_persistent=True, initiator_name=None):
    session_id = iscsi_struct.ISCSI_UNIQUE_SESSION_ID()
    connection_id = iscsi_struct.ISCSI_UNIQUE_CONNECTION_ID()
    portal_ref = ctypes.byref(portal) if portal else None
    login_opts_ref = ctypes.byref(login_opts) if login_opts else None
    initiator_name_ref = ctypes.c_wchar_p(initiator_name) if initiator_name else None
    self._run_and_check_output(iscsidsc.LoginIScsiTargetW, ctypes.c_wchar_p(target_name), False, initiator_name_ref, ctypes.c_ulong(w_const.ISCSI_ANY_INITIATOR_PORT), portal_ref, iscsi_struct.ISCSI_SECURITY_FLAGS(), None, login_opts_ref, ctypes.c_ulong(0), None, is_persistent, ctypes.byref(session_id), ctypes.byref(connection_id), ignored_error_codes=[w_const.ISDSC_TARGET_ALREADY_LOGGED_IN])
    return (session_id, connection_id)