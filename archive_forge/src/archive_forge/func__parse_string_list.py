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
@staticmethod
def _parse_string_list(buff, element_count):
    buff = ctypes.cast(buff, ctypes.POINTER(ctypes.c_wchar))
    str_list = buff[:element_count].strip('\x00')
    str_list = str_list.split('\x00') if str_list else []
    return str_list