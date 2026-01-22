import ctypes
import struct
from eventlet import patcher
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def _wait_io_completion(self, event):
    self._run_and_check_output(kernel32.WaitForSingleObjectEx, event, WAIT_INFINITE_TIMEOUT, True, error_ret_vals=[w_const.WAIT_FAILED])