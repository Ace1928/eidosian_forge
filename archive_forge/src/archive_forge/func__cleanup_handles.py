import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _cleanup_handles(self):
    self._close_pipe()
    if self._log_file_handle:
        self._log_file_handle.close()
        self._log_file_handle = None
    if self._r_overlapped.hEvent:
        self._ioutils.close_handle(self._r_overlapped.hEvent)
        self._r_overlapped.hEvent = None
    if self._w_overlapped.hEvent:
        self._ioutils.close_handle(self._w_overlapped.hEvent)
        self._w_overlapped.hEvent = None