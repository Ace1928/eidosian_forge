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
@_utils.retry_decorator(exceptions=exceptions.Win32IOException, max_sleep_time=2)
def wait_named_pipe(self, pipe_name, timeout=WAIT_PIPE_DEFAULT_TIMEOUT):
    """Wait a given amount of time for a pipe to become available."""
    self._run_and_check_output(kernel32.WaitNamedPipeW, ctypes.c_wchar_p(pipe_name), timeout * units.k)