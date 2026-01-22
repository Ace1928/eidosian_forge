import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _close_pipe(self):
    if self._pipe_handle:
        self._ioutils.close_handle(self._pipe_handle)
        self._pipe_handle = None